"""KNN aggregation of donor scores over content neighbors (naive method).

For a cold item we take the k nearest (by content) warm items; the score is the
similarity-weighted average of donor scores for those neighbors. This reproduces the naive
approach, which usually loses to Grouped MP (it pulls in the global popularity of neighbors).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer._pdutils import map_codes, unique_sorted
from warmtransfer.columns import Columns as C
from warmtransfer.methods.base import ColdStartMethod, cross_join_frame, register_method
from warmtransfer.types import TransferInputs


@register_method("knn_score_avg")
class KNNScoreAggregation(ColdStartMethod):
    """Similarity-weighted average of donor scores over k content neighbors.

    :param k: number of nearest warm neighbors.
    :param clip_negative: zero out negative similarities (cosine can be < 0).
    """

    requires = frozenset({"donor_scores", "similarity", "content"})

    def __init__(self, k: int = 20, clip_negative: bool = True) -> None:
        super().__init__()
        self.k = k
        self.clip_negative = clip_negative

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        if inputs.warm_features is None or inputs.cold_features is None:
            raise ValueError("knn_score_avg requires warm_features and cold_features")
        if inputs.similarity is None:
            raise ValueError("knn_score_avg requires similarity [n_cold, n_warm]")

        self._warm_ids = np.asarray(inputs.warm_features.item_ids)
        self._cold_ids = np.asarray(inputs.cold_features.item_ids)
        self._cold_pos = {it: r for r, it in enumerate(self._cold_ids)}

        # donor score matrix: [n_users, n_warm], aligned to self._warm_ids
        self._donor_matrix, self._user_ids = _pivot_scores(inputs.donor_scores, self._warm_ids)
        self._user_pos = {u: i for i, u in enumerate(self._user_ids)}

        # for each cold item: indices of top-k neighbors and normalized weights
        sim = np.asarray(inputs.similarity, dtype=float)
        if self.clip_negative:
            sim = np.clip(sim, 0.0, None)
        self._neighbors, self._weights = _topk_neighbors(sim, self.k)

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        user_ids = np.asarray(user_ids)
        cold_item_ids = np.asarray(cold_item_ids)

        # donor-matrix rows for the requested users (unknown ones -> zeros)
        rows = np.array([self._user_pos.get(u, -1) for u in user_ids])
        known = rows >= 0
        donor = np.zeros((len(user_ids), self._donor_matrix.shape[1]))
        donor[known] = self._donor_matrix[rows[known]]

        scores = np.zeros((len(user_ids), len(cold_item_ids)))
        for c, item in enumerate(cold_item_ids):
            r = self._cold_pos.get(item)
            if r is None:
                continue
            nb = self._neighbors[r]
            w = self._weights[r]
            # [n_users] = donor[:, neighbors] @ weights
            scores[:, c] = donor[:, nb] @ w
        return cross_join_frame(user_ids, cold_item_ids, scores)

    def get_params(self) -> dict:
        return {"k": self.k, "clip_negative": self.clip_negative}


def _pivot_scores(
    donor_scores: pd.DataFrame, warm_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pivot long-format donor scores into a [n_users, n_warm] matrix aligned to ``warm_ids``."""
    w_pos = {it: j for j, it in enumerate(warm_ids)}
    user_ids = unique_sorted(donor_scores[C.User])
    u_pos = {u: i for i, u in enumerate(user_ids)}

    # donor scores are defined over warm items -> every item_id is present in w_pos
    matrix = np.zeros((len(user_ids), len(warm_ids)))
    rows = map_codes(donor_scores[C.User], u_pos)
    cols = map_codes(donor_scores[C.Item], w_pos)
    matrix[rows, cols] = donor_scores[C.Score].to_numpy()
    return matrix, user_ids


def _topk_neighbors(sim: np.ndarray, k: int) -> tuple[list, list]:
    """For each row of sim, return top-k indices and normalized weights (sum = 1)."""
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
