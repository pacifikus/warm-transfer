"""Donor: Bayesian Personalized Ranking (implicit library).

Pairwise ranking donor (unlike the pointwise ALS): it optimizes the ordering of warm
items for each user. The interface matches ALS — scores are the dot product of factors.
Needed as a second MF donor to test donor-agnosticism across different losses.
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


@register_adapter("bpr")
class BPRAdapter(ModelAdapter):
    """implicit BayesianPersonalizedRanking.

    :param factors: dimensionality of the latent space.
    :param learning_rate: SGD step size.
    :param regularization: L2 regularization.
    :param iterations: number of epochs.
    """

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        iterations: int = 100,
    ) -> None:
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self._user_ids: np.ndarray | None = None
        self._item_ids: np.ndarray | None = None

    def fit(self, dataset: Dataset, seed: int = 0) -> BPRAdapter:
        from threadpoolctl import threadpool_limits

        set_global_seed(seed)
        inter = validate_interactions(dataset.interactions)

        self._user_ids = unique_sorted(inter[C.User])
        self._item_ids = unique_sorted(inter[C.Item])
        self._u_pos = {u: i for i, u in enumerate(self._user_ids)}
        self._i_pos = {it: j for j, it in enumerate(self._item_ids)}

        rows = map_codes(inter[C.User], self._u_pos)
        cols = map_codes(inter[C.Item], self._i_pos)
        vals = inter[C.Weight].to_numpy().astype(np.float32)
        user_items = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(len(self._user_ids), len(self._item_ids))
        )

        with threadpool_limits(1, "blas"):
            from implicit.bpr import BayesianPersonalizedRanking

            self._model = BayesianPersonalizedRanking(
                factors=self.factors,
                learning_rate=self.learning_rate,
                regularization=self.regularization,
                iterations=self.iterations,
                random_state=seed,
            )
            self._model.fit(user_items, show_progress=False)
        return self

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> pd.DataFrame:
        """Scores for the cross product user_ids × item_ids (only known warm items)."""
        if self._user_ids is None or self._item_ids is None:
            raise RuntimeError("BPRAdapter: call fit() before score()")

        u_known = [u for u in user_ids if u in self._u_pos]
        i_known = [it for it in item_ids if it in self._i_pos]
        u_idx = np.array([self._u_pos[u] for u in u_known], dtype=int)
        i_idx = np.array([self._i_pos[it] for it in i_known], dtype=int)

        # implicit BPR keeps the bias in an extra factor column (user column=1, item column=bias);
        # the full dot product yields a correct score that accounts for the bias.
        uf = np.asarray(self._model.user_factors[u_idx])
        itf = np.asarray(self._model.item_factors[i_idx])
        scores = uf @ itf.T  # [n_u, n_i]

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
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "iterations": self.iterations,
        }
