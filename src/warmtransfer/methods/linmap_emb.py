"""Отображение контента айтема в латентные факторы донора (Gantner, ICDM 2010).

Embedding-версия ``linmap``: вместо ``контент → скор`` учим Ridge ``контент → эмбеддинг
донора``. По warm-айтемам известны и контент, и факторы донора — обучаем регрессию
``content → item_factors``. Для cold-айтема применяем её к контенту, получаем оценку его
вектора факторов, затем скор пары (user, cold_item) = ``user_emb · cold_emb``.

Отличие от ``linmap`` (контент→скор): целевая переменная — латентные факторы, а не вектор
скоров по пользователям. Это классический attribute-to-feature mapping; нужен доступ к
эмбеддингам донора ([EMB]).

Ссылка: Gantner et al., "Learning Attribute-to-Feature Mappings for Cold-Start
Recommendations", ICDM 2010.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from warmtransfer.methods.base import ColdStartMethod, cross_join_frame, register_method
from warmtransfer.methods.linmap import _dense
from warmtransfer.types import TransferInputs


@register_method("linmap_emb")
class LinMapEmbedding(ColdStartMethod):
    """Ridge-отображение контент → латентные факторы донора (Gantner).

    :param alpha: коэффициент L2-регуляризации Ridge.
    """

    requires = frozenset({"embeddings", "content"})

    def __init__(self, alpha: float = 10.0) -> None:
        super().__init__()
        self.alpha = alpha

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        if inputs.warm_features is None or inputs.cold_features is None:
            raise ValueError(f"{self.name} требует warm_features и cold_features")
        if inputs.embeddings is None:
            raise ValueError(f"{self.name} требует embeddings")
        emb = inputs.embeddings
        for key in ("item", "item_ids", "user", "user_ids"):
            if key not in emb:
                raise ValueError(f"embeddings: отсутствует ключ {key!r}")

        item_emb = np.asarray(emb["item"], dtype=float)  # [m, d]
        item_emb_ids = np.asarray(emb["item_ids"])  # [m]
        self._user_emb = np.asarray(emb["user"], dtype=float)  # [p, d]
        user_emb_ids = np.asarray(emb["user_ids"])  # [p]
        self._user_pos = {u: i for i, u in enumerate(user_emb_ids)}

        warm_ids = np.asarray(inputs.warm_features.item_ids)
        emb_pos = {it: j for j, it in enumerate(item_emb_ids)}
        # оставляем только warm-айтемы, у которых есть и контент, и эмбеддинг
        keep = np.array([it in emb_pos for it in warm_ids])
        if not keep.any():
            raise ValueError(f"{self.name}: нет warm-айтемов с контентом и эмбеддингом")
        warm_kept = warm_ids[keep]
        emb_rows = np.array([emb_pos[it] for it in warm_kept])

        x_warm = np.asarray(_dense(inputs.warm_features.subset(warm_kept).matrix), dtype=float)
        y_warm = item_emb[emb_rows]  # [n_warm_kept, d]

        self._model = Ridge(alpha=self.alpha)
        self._model.fit(x_warm, y_warm)

        # средняя магнитуда warm-эмбеддингов — нужна наследнику (magnitude scaling)
        self._warm_mean_norm = float(np.linalg.norm(y_warm, axis=1).mean())

        # оценка факторов всех cold-айтемов [n_cold, d]
        self._cold_ids = np.asarray(inputs.cold_features.item_ids)
        self._cold_pos = {it: r for r, it in enumerate(self._cold_ids)}
        x_cold = np.asarray(_dense(inputs.cold_features.matrix), dtype=float)
        self._cold_emb = np.asarray(self._model.predict(x_cold))
        self._postprocess_cold_emb()

    def _postprocess_cold_emb(self) -> None:
        """Хук для наследников (magnitude scaling). По умолчанию ничего не делает."""

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        user_ids = np.asarray(user_ids)
        cold_item_ids = np.asarray(cold_item_ids)

        rows = np.array([self._user_pos.get(u, -1) for u in user_ids])
        known = rows >= 0
        u_emb = np.zeros((len(user_ids), self._user_emb.shape[1]))
        u_emb[known] = self._user_emb[rows[known]]

        c_rows = np.array([self._cold_pos.get(it, -1) for it in cold_item_ids])
        c_known = c_rows >= 0
        c_emb = np.zeros((len(cold_item_ids), self._cold_emb.shape[1]))
        c_emb[c_known] = self._cold_emb[c_rows[c_known]]

        scores = u_emb @ c_emb.T  # [n_users, n_cold]
        return cross_join_frame(user_ids, cold_item_ids, scores)

    def get_params(self) -> dict:
        return {"alpha": self.alpha}
