"""Донор: матричная факторизация ALS (библиотека implicit).

Обучается на warm-взаимодействиях, выдаёт скоры по парам (user, warm_item) как
скалярное произведение факторов. Cold-айтемов в обучении нет (их факторов не существует).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from warmtransfer._pdutils import map_codes, unique_sorted
from warmtransfer.bench.adapters.base import ModelAdapter, register_adapter
from warmtransfer.columns import Columns as C
from warmtransfer.schema import validate_interactions
from warmtransfer.seeding import set_global_seed
from warmtransfer.types import Dataset


@register_adapter("als")
class ALSAdapter(ModelAdapter):
    """implicit AlternatingLeastSquares.

    :param factors: размерность латентного пространства.
    :param regularization: L2-регуляризация.
    :param iterations: число итераций ALS.
    :param alpha: множитель confidence для весов взаимодействий.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        iterations: int = 20,
        alpha: float = 1.0,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self._user_ids: np.ndarray | None = None
        self._item_ids: np.ndarray | None = None

    def fit(self, dataset: Dataset, seed: int = 0) -> ALSAdapter:
        from threadpoolctl import threadpool_limits

        set_global_seed(seed)
        inter = validate_interactions(dataset.interactions)

        self._user_ids = unique_sorted(inter[C.User])
        self._item_ids = unique_sorted(inter[C.Item])
        self._u_pos = {u: i for i, u in enumerate(self._user_ids)}
        self._i_pos = {it: j for j, it in enumerate(self._item_ids)}

        rows = map_codes(inter[C.User], self._u_pos)
        cols = map_codes(inter[C.Item], self._i_pos)
        vals = (inter[C.Weight].to_numpy() * self.alpha).astype(np.float32)
        user_items = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(len(self._user_ids), len(self._item_ids))
        )

        # OpenBLAS-threadpool конфликтует с внутренним параллелизмом implicit
        with threadpool_limits(1, "blas"):
            from implicit.als import AlternatingLeastSquares

            self._model = AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                random_state=seed,
            )
            self._model.fit(user_items, show_progress=False)
        return self

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> pd.DataFrame:
        """Скоры для кросс-произведения user_ids × item_ids (только известные warm)."""
        if self._user_ids is None or self._item_ids is None:
            raise RuntimeError("ALSAdapter: вызовите fit() до score()")

        u_known = [u for u in user_ids if u in self._u_pos]
        i_known = [it for it in item_ids if it in self._i_pos]
        u_idx = np.array([self._u_pos[u] for u in u_known], dtype=int)
        i_idx = np.array([self._i_pos[it] for it in i_known], dtype=int)

        uf = self._model.user_factors[u_idx]
        itf = self._model.item_factors[i_idx]
        scores = np.asarray(uf) @ np.asarray(itf).T  # [n_u, n_i]

        return pd.DataFrame(
            {
                C.User: np.repeat(u_known, len(i_known)),
                C.Item: np.tile(i_known, len(u_known)),
                C.Score: scores.reshape(-1),
            }
        )

    def embeddings(self) -> dict[str, np.ndarray] | None:
        if self._user_ids is None or self._item_ids is None:
            return None
        return {
            "user": np.asarray(self._model.user_factors),
            "item": np.asarray(self._model.item_factors),
            "user_ids": self._user_ids,
            "item_ids": self._item_ids,
        }

    def get_params(self) -> dict:
        return {
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "alpha": self.alpha,
        }
