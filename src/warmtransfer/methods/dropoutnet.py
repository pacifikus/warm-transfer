"""DropoutNet (Volkovs et al., NeurIPS 2017) — post-hoc cold-start via preference dropout.

Idea: we train an MLP that, given the input [donor item latent, item content], reconstructs
the item latent. During training, the latent (preference) branch is randomly zeroed out
(dropout) — this mimics cold-start and forces the network to learn the mapping
"content -> latent". At inference, a cold item has no latent (=0), so the score is computed
entirely from content:
``score(u, i) = user_emb[u] · MLP([0, content_i])``.

Unlike the linear LinMap (content -> score vector), DropoutNet predicts the item latent in
the donor space and is non-linear. It requires access to the donor embeddings ([EMB]).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.methods.base import ColdStartMethod, cross_join_frame, register_method
from warmtransfer.types import TransferInputs


@register_method("dropoutnet")
class DropoutNet(ColdStartMethod):
    """MLP [latent⊕content]->latent with preference dropout (item embedding reconstruction).

    :param hidden: hidden layer size.
    :param epochs: number of training epochs.
    :param lr: learning rate (Adam).
    :param dropout_pref: fraction of examples in the batch with a zeroed latent branch.
    :param weight_decay: L2 regularization.
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
            raise ValueError("dropoutnet requires warm_features and cold_features")
        if inputs.embeddings is None:
            raise ValueError("dropoutnet requires embeddings")
        emb = inputs.embeddings
        for key in ("item", "item_ids", "user", "user_ids"):
            if key not in emb:
                raise ValueError(f"embeddings: missing key {key!r}")

        torch.manual_seed(seed)

        item_emb = np.asarray(emb["item"], dtype=np.float32)  # [m, d]
        item_emb_ids = np.asarray(emb["item_ids"])
        self._user_emb = np.asarray(emb["user"], dtype=np.float32)  # [p, d]
        user_ids = np.asarray(emb["user_ids"])
        self._user_pos = {u: i for i, u in enumerate(user_ids)}

        # warm items that have BOTH a donor embedding AND content
        warm = inputs.warm_features
        warm_mat = np.asarray(_dense(warm.matrix), dtype=np.float32)  # [n_warm, c]
        emb_pos = {it: j for j, it in enumerate(item_emb_ids)}
        warm_rows = [r for r, it in enumerate(warm.item_ids) if it in emb_pos]
        if not warm_rows:
            raise ValueError("dropoutnet: no warm items with a donor embedding")
        emb_rows = [emb_pos[warm.item_ids[r]] for r in warm_rows]

        x = torch.from_numpy(warm_mat[warm_rows])  # [n, c] content
        v = torch.from_numpy(item_emb[emb_rows])  # [n, d] latent target

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
            # preference dropout: some rows get a zeroed latent branch
            mask = (torch.rand(v.shape[0], 1, generator=gen) >= self.dropout_pref).float()
            inp = torch.cat([v * mask, x], dim=1)
            pred = self._net(inp)
            loss = loss_fn(pred, v)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # latent of cold items: preferences = 0, content only
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
    """Convert features to a dense ndarray (sparse -> toarray)."""
    todense = getattr(matrix, "toarray", None)
    if callable(todense):
        return np.asarray(todense())
    return np.asarray(matrix)
