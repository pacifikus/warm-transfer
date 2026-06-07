"""DropoutNet (Volkovs et al., NeurIPS 2017) — post-hoc cold-start через dropout преференций.

Идея: обучаем MLP, который по входу [item-латент донора, контент айтема] восстанавливает
item-латент. Во время обучения латентную (preference) ветку случайно зануляем (dropout) — это
имитирует cold-start и заставляет сеть выучить отображение «контент → латент». На инференсе у
cold-айтема латента нет (=0), поэтому скор считается полностью по контенту:
``score(u, i) = user_emb[u] · MLP([0, content_i])``.

В отличие от линейного LinMap (контент → вектор скоров), DropoutNet предсказывает item-латент
в пространстве донора и нелинеен. Требует доступа к эмбеддингам донора ([EMB]).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.methods.base import ColdStartMethod, cross_join_frame, register_method
from warmtransfer.types import TransferInputs


@register_method("dropoutnet")
class DropoutNet(ColdStartMethod):
    """MLP [латент⊕контент]→латент с dropout преференций (восстановление item-эмбеддинга).

    :param hidden: размер скрытого слоя.
    :param epochs: число эпох обучения.
    :param lr: learning rate (Adam).
    :param dropout_pref: доля примеров в батче с занулённой латентной веткой.
    :param weight_decay: L2-регуляризация.
    """

    requires = frozenset({"embeddings", "content"})

    def __init__(
        self,
        hidden: int = 128,
        epochs: int = 200,
        lr: float = 1e-3,
        dropout_pref: float = 0.5,
        weight_decay: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.dropout_pref = dropout_pref
        self.weight_decay = weight_decay

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        import torch
        from torch import nn

        if inputs.warm_features is None or inputs.cold_features is None:
            raise ValueError("dropoutnet требует warm_features и cold_features")
        if inputs.embeddings is None:
            raise ValueError("dropoutnet требует embeddings")
        emb = inputs.embeddings
        for key in ("item", "item_ids", "user", "user_ids"):
            if key not in emb:
                raise ValueError(f"embeddings: отсутствует ключ {key!r}")

        torch.manual_seed(seed)

        item_emb = np.asarray(emb["item"], dtype=np.float32)  # [m, d]
        item_emb_ids = np.asarray(emb["item_ids"])
        self._user_emb = np.asarray(emb["user"], dtype=np.float32)  # [p, d]
        user_ids = np.asarray(emb["user_ids"])
        self._user_pos = {u: i for i, u in enumerate(user_ids)}

        # warm-айтемы, у которых есть И эмбеддинг донора, И контент
        warm = inputs.warm_features
        warm_mat = np.asarray(_dense(warm.matrix), dtype=np.float32)  # [n_warm, c]
        emb_pos = {it: j for j, it in enumerate(item_emb_ids)}
        warm_rows = [r for r, it in enumerate(warm.item_ids) if it in emb_pos]
        if not warm_rows:
            raise ValueError("dropoutnet: нет warm-айтемов с эмбеддингом донора")
        emb_rows = [emb_pos[warm.item_ids[r]] for r in warm_rows]

        x = torch.from_numpy(warm_mat[warm_rows])  # [n, c] контент
        v = torch.from_numpy(item_emb[emb_rows])  # [n, d] латент-таргет

        d = v.shape[1]
        c = x.shape[1]
        self._d = d
        self._net = nn.Sequential(
            nn.Linear(d + c, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, d),
        )
        opt = torch.optim.Adam(
            self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(seed)

        self._net.train()
        for _ in range(self.epochs):
            # dropout преференций: часть строк получает нулевую латентную ветку
            mask = (torch.rand(v.shape[0], 1, generator=gen) >= self.dropout_pref).float()
            inp = torch.cat([v * mask, x], dim=1)
            pred = self._net(inp)
            loss = loss_fn(pred, v)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # латент cold-айтемов: преференции = 0, только контент
        self._net.eval()
        cold = inputs.cold_features
        cold_mat = np.asarray(_dense(cold.matrix), dtype=np.float32)
        self._cold_pos = {it: r for r, it in enumerate(cold.item_ids)}
        with torch.no_grad():
            zeros = torch.zeros(cold_mat.shape[0], d)
            inp = torch.cat([zeros, torch.from_numpy(cold_mat)], dim=1)
            self._cold_emb = self._net(inp).numpy()  # [n_cold, d]

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        user_ids = np.asarray(user_ids)
        cold_item_ids = np.asarray(cold_item_ids)

        rows = np.array([self._user_pos.get(u, -1) for u in user_ids])
        known = rows >= 0
        u_emb = np.zeros((len(user_ids), self._d), dtype=np.float32)
        u_emb[known] = self._user_emb[rows[known]]

        c_rows = np.array([self._cold_pos.get(it, -1) for it in cold_item_ids])
        c_known = c_rows >= 0
        c_emb = np.zeros((len(cold_item_ids), self._d), dtype=np.float32)
        c_emb[c_known] = self._cold_emb[c_rows[c_known]]

        scores = u_emb @ c_emb.T  # [n_users, n_cold]
        return cross_join_frame(user_ids, cold_item_ids, scores)

    def get_params(self) -> dict:
        return {
            "hidden": self.hidden,
            "epochs": self.epochs,
            "lr": self.lr,
            "dropout_pref": self.dropout_pref,
            "weight_decay": self.weight_decay,
        }


def _dense(matrix: object) -> np.ndarray:
    """Привести признаки к плотному ndarray (sparse → toarray)."""
    todense = getattr(matrix, "toarray", None)
    if callable(todense):
        return np.asarray(todense())
    return np.asarray(matrix)
