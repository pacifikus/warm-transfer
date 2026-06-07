"""KNN aggregation of donor scores with neighbor popularity debiasing.

Like the naive KNN (see ``knn.py``), but each warm neighbor's per-user mean score
(a popularity proxy) is subtracted from its score. This reduces the contribution of
globally popular neighbors: they pull recommendations toward "what everyone likes"
rather than the user's personal preferences.

    score(u, i) = Σ_j w_j · (donor[u, j] - colmean_j),

where ``colmean_j`` is the mean of donor[:, j] over users, and ``w_j`` is the
sum-normalized content similarity (as in knn.py, clip_negative=True).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer._pdutils import map_codes, unique_sorted
from warmtransfer.columns import Columns as C
from warmtransfer.methods.base import ColdStartMethod, cross_join_frame, register_method
from warmtransfer.types import TransferInputs


@register_method("debiased_knn")
class DebiasedKNN(ColdStartMethod):
    """KNN over content neighbors with subtraction of neighbor popularity.

    :param k: number of nearest warm neighbors.
    :param clip_negative: zero out negative similarities (cosine can be <0).
    """

    requires = frozenset({"donor_scores", "similarity", "content"})

    def __init__(self, k: int = 20, clip_negative: bool = True) -> None:
        super().__init__()
        self.k = k
        self.clip_negative = clip_negative

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        if inputs.warm_features is None or inputs.cold_features is None:
            raise ValueError("debiased_knn requires warm_features and cold_features")
        if inputs.similarity is None:
            raise ValueError("debiased_knn requires similarity [n_cold, n_warm]")

        self._warm_ids = np.asarray(inputs.warm_features.item_ids)
        self._cold_ids = np.asarray(inputs.cold_features.item_ids)
        self._cold_pos = {it: r for r, it in enumerate(self._cold_ids)}

        # donor score matrix: [n_users, n_warm], aligned with self._warm_ids
        self._donor_matrix, self._user_ids = _pivot_scores(inputs.donor_scores, self._warm_ids)
        self._user_pos = {u: i for i, u in enumerate(self._user_ids)}

        # neighbor popularity proxy: per-user mean score of the column [n_warm]
        self._col_mean = self._donor_matrix.mean(axis=0)
        # debiased matrix: the neighbor popularity is subtracted from the score
        self._debiased = self._donor_matrix - self._col_mean

        # for each cold item: indices of the top-k neighbors and normalized weights
        sim = np.asarray(inputs.similarity, dtype=float)
        if self.clip_negative:
            sim = np.clip(sim, 0.0, None)
        self._neighbors, self._weights = _topk_neighbors(sim, self.k)

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        user_ids = np.asarray(user_ids)
        cold_item_ids = np.asarray(cold_item_ids)

        # rows of the debiased matrix for the requested users
        # unknown user → zero row (no personal deviations)
        rows = np.array([self._user_pos.get(u, -1) for u in user_ids])
        known = rows >= 0
        debiased = np.zeros((len(user_ids), self._debiased.shape[1]))
        debiased[known] = self._debiased[rows[known]]

        scores = np.zeros((len(user_ids), len(cold_item_ids)))
        for c, item in enumerate(cold_item_ids):
            r = self._cold_pos.get(item)
            if r is None:
                continue
            nb = self._neighbors[r]
            w = self._weights[r]
            # [n_users] = debiased[:, neighbors] @ weights
            scores[:, c] = debiased[:, nb] @ w
        return cross_join_frame(user_ids, cold_item_ids, scores)

    def get_params(self) -> dict:
        return {"k": self.k, "clip_negative": self.clip_negative}


def _pivot_scores(
    donor_scores: pd.DataFrame, warm_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pivot long-format donor scores into a [n_users, n_warm] matrix aligned with ``warm_ids``."""
    w_pos = {it: j for j, it in enumerate(warm_ids)}
    user_ids = unique_sorted(donor_scores[C.User])
    u_pos = {u: i for i, u in enumerate(user_ids)}

    # donor scores are defined over warm items → every item_id is present in w_pos
    matrix = np.zeros((len(user_ids), len(warm_ids)))
    rows = map_codes(donor_scores[C.User], u_pos)
    cols = map_codes(donor_scores[C.Item], w_pos)
    matrix[rows, cols] = donor_scores[C.Score].to_numpy()
    return matrix, user_ids


def _topk_neighbors(sim: np.ndarray, k: int) -> tuple[list, list]:
    """For each row of sim, return the top-k indices and normalized weights (Σ=1)."""
    n_warm = sim.shape[1]
    kk = min(k, n_warm)
    neighbors: list = []
    weights: list = []
    for row in sim:
        idx = np.argpartition(-row, kk - 1)[:kk]
        idx = idx[np.argsort(-row[idx])]
        w = row[idx]
        total = w.sum()
        w = w / total if total > 0 else np.full(kk, 1.0 / kk)
        neighbors.append(idx)
        weights.append(w)
    return neighbors, weights
