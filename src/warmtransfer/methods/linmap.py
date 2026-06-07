"""Linear mapping from item content to its donor score vector (Ridge).

Model-agnostic method: we train a Ridge regression that, given the content vector of a warm
item, predicts its donor scores across all users at once (multi-output). For a cold item we
apply the same regression to its content and obtain scores across all users.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from warmtransfer._pdutils import map_codes, unique_sorted
from warmtransfer.columns import Columns as C
from warmtransfer.methods.base import ColdStartMethod, cross_join_frame, register_method
from warmtransfer.types import ItemFeatures, TransferInputs


@register_method("linmap")
class LinMap(ColdStartMethod):
    """Ridge mapping content -> donor score vector across all users.

    :param alpha: Ridge L2 regularization coefficient.
    """

    requires = frozenset({"donor_scores", "content"})

    def __init__(self, alpha: float = 10.0) -> None:
        super().__init__()
        self.alpha = alpha

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        if inputs.warm_features is None or inputs.cold_features is None:
            raise ValueError("linmap requires warm_features and cold_features")

        warm_ids = np.asarray(inputs.warm_features.item_ids)

        # target matrix S [n_warm, n_users]: rows = warm items, columns = users
        targets, self._user_ids = _pivot_scores_t(inputs.donor_scores, warm_ids)
        self._user_pos = {u: i for i, u in enumerate(self._user_ids)}

        # warm item features [n_warm, n_feat] (aligned to warm_ids)
        x = np.asarray(_dense(inputs.warm_features.matrix), dtype=float)

        self._model = Ridge(alpha=self.alpha)
        self._model.fit(x, targets)
        self._cold: ItemFeatures = inputs.cold_features

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        user_ids = np.asarray(user_ids)
        cold_item_ids = np.asarray(cold_item_ids)

        # content of the requested cold items [n_cold, n_feat]
        x_cold = np.asarray(_dense(self._cold.subset(cold_item_ids).matrix), dtype=float)
        # score prediction [n_cold, n_users_all]
        pred = np.asarray(self._model.predict(x_cold))

        # pick columns for the requested users (unknown -> zeros)
        cols = np.array([self._user_pos.get(u, -1) for u in user_ids])
        known = cols >= 0
        # result [n_users, n_cold]
        scores = np.zeros((len(user_ids), len(cold_item_ids)))
        scores[known] = pred[:, cols[known]].T
        return cross_join_frame(user_ids, cold_item_ids, scores)

    def get_params(self) -> dict:
        return {"alpha": self.alpha}


def _pivot_scores_t(
    donor_scores: pd.DataFrame, warm_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pivot donor scores into a matrix [n_warm, n_users] aligned to ``warm_ids``."""
    w_pos = {it: i for i, it in enumerate(warm_ids)}
    extra = set(unique_sorted(donor_scores[C.Item]).tolist()) - set(w_pos)
    if extra:
        raise ValueError(
            f"donor_scores contain non-warm items ({len(extra)} of them) — violation of "
            "the warm-only contract: linmap must train only on warm scores"
        )
    user_ids = unique_sorted(donor_scores[C.User])
    u_pos = {u: j for j, u in enumerate(user_ids)}

    matrix = np.zeros((len(warm_ids), len(user_ids)))
    rows = map_codes(donor_scores[C.Item], w_pos)
    cols = map_codes(donor_scores[C.User], u_pos)
    matrix[rows, cols] = donor_scores[C.Score].to_numpy()
    return matrix, user_ids


def _dense(matrix: object) -> np.ndarray:
    """Convert features to a dense ndarray (sparse -> toarray)."""
    todense = getattr(matrix, "toarray", None)
    if callable(todense):
        return np.asarray(todense())
    return np.asarray(matrix)
