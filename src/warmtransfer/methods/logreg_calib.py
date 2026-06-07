"""LogReg calibration of donor scores via content neighbors.

For a cold item we take the k nearest warm neighbors by content and compute several
aggregates over their donor scores (weighted, mean, max). A logistic regression trained on
the val-cold fold calibrates these aggregates into an interaction probability. Unlike naive
KNN, the aggregate weights are fitted to the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer._pdutils import map_codes, unique_sorted
from warmtransfer.columns import Columns as C
from warmtransfer.methods.base import ColdStartMethod, cross_join_frame, register_method
from warmtransfer.types import ItemFeatures, TransferInputs


@register_method("logreg_calib")
class LogRegCalibration(ColdStartMethod):
    """Logistic calibration of knn aggregates of donor scores (trained on val-cold).

    :param k: number of content neighbors.
    """

    requires = frozenset({"donor_scores", "similarity", "content", "val"})

    def __init__(self, k: int = 20) -> None:
        super().__init__()
        self.k = k

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        warm = _req(inputs.warm_features, "warm_features")
        cold = _req(inputs.cold_features, "cold_features")
        val_cold = _req(inputs.val_cold_features, "val_cold_features")
        if inputs.similarity is None or inputs.val_similarity is None:
            raise ValueError("logreg_calib requires similarity and val_similarity")
        if inputs.val_interactions is None:
            raise ValueError("logreg_calib requires val_interactions")

        self._users = unique_sorted(inputs.donor_scores[C.User])
        self._u_pos = {u: i for i, u in enumerate(self._users)}
        self._donor = _pivot_donor(inputs.donor_scores, warm.item_ids, self._users, self._u_pos)

        self._cold = cold
        self._cold_sim = np.asarray(inputs.similarity, dtype=float)
        self._cold_pos = {it: r for r, it in enumerate(cold.item_ids)}

        x_val = self._features(np.asarray(inputs.val_similarity, dtype=float), self._users)
        y_val = _labels(inputs.val_interactions, self._users, val_cold.item_ids)

        self._scaler = StandardScaler().fit(x_val)
        self._meta = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._meta.fit(self._scaler.transform(x_val), y_val)

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        user_ids = np.asarray(user_ids)
        cold_item_ids = np.asarray(cold_item_ids)

        rows = [self._cold_pos[it] for it in cold_item_ids]
        sim = self._cold_sim[rows]
        x = self._features(sim, user_ids)
        proba = self._meta.predict_proba(self._scaler.transform(x))[:, 1]
        scores = proba.reshape(len(user_ids), len(cold_item_ids))
        return cross_join_frame(user_ids, cold_item_ids, scores)

    def _features(self, sim: np.ndarray, users: np.ndarray) -> np.ndarray:
        """Donor score aggregates over top-k neighbors: [weighted, mean, max]."""
        u_rows = np.array([self._u_pos.get(u, -1) for u in users])
        known = u_rows >= 0
        donor = np.zeros((len(users), self._donor.shape[1]))
        donor[known] = self._donor[u_rows[known]]

        n_items, n_warm = sim.shape
        kk = min(self.k, n_warm)
        sim = np.clip(sim, 0.0, None)
        weighted = np.zeros((len(users), n_items))
        mean = np.zeros((len(users), n_items))
        mx = np.zeros((len(users), n_items))
        for c in range(n_items):
            idx = np.argpartition(-sim[c], kk - 1)[:kk]
            w = sim[c][idx]
            total = w.sum()
            w = w / total if total > 0 else np.full(kk, 1.0 / kk)
            block = donor[:, idx]  # [n_users, kk]
            weighted[:, c] = block @ w
            mean[:, c] = block.mean(axis=1)
            mx[:, c] = block.max(axis=1)
        return np.column_stack([weighted.reshape(-1), mean.reshape(-1), mx.reshape(-1)])

    def get_params(self) -> dict:
        return {"k": self.k}


def _req(feats: ItemFeatures | None, what: str) -> ItemFeatures:
    if feats is None:
        raise ValueError(f"logreg_calib requires {what}")
    return feats


def _pivot_donor(
    donor_scores: pd.DataFrame, warm_ids: np.ndarray, users: np.ndarray, u_pos: dict
) -> np.ndarray:
    w_pos = {it: j for j, it in enumerate(warm_ids)}
    matrix = np.zeros((len(users), len(warm_ids)))
    rows = map_codes(donor_scores[C.User], u_pos)
    cols = map_codes(donor_scores[C.Item], w_pos)
    matrix[rows, cols] = donor_scores[C.Score].to_numpy()
    return matrix


def _labels(val_interactions: pd.DataFrame, users: np.ndarray, items: np.ndarray) -> np.ndarray:
    pos = set(zip(val_interactions[C.User], val_interactions[C.Item], strict=True))
    labels = np.zeros((len(users), len(items)), dtype=int)
    for ui, u in enumerate(users):
        for ci, it in enumerate(items):
            if (u, it) in pos:
                labels[ui, ci] = 1
    return labels.reshape(-1)
