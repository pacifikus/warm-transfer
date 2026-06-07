"""Stacking+: meta-model over the strong LinMap signal + personalized popularity.

Motivation (from the benchmark): LinMap wins on ranking, but in popularity-
dominated domains (Goodbooks) the personalized Grouped MP is stronger in top-k. The base
``stacking`` uses the donor's weak knn signal and falls short of LinMap. Here a meta logreg
is trained over three signals:

  1. ``linmap`` — Ridge content→donor scores (strong ranking, transfer of latent structure);
  2. ``genre affinity`` — user's affinity to the item's genres (strong in top-k on popular domains);
  3. ``genre popularity`` — global genre popularity (baseline used as a feature).

Weights are fit by logistic regression on the val-cold fold (interaction fact as target),
so the method adaptively takes the best of both worlds and should be robust across domains.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from warmtransfer.methods.base import ColdStartMethod, cross_join_frame, register_method
from warmtransfer.methods.linmap import _dense, _pivot_scores_t
from warmtransfer.methods.stacking import _labels, _user_genre_affinity, _warm_popularity
from warmtransfer.types import ItemFeatures, TransferInputs


@register_method("stacking_plus")
class StackingPlus(ColdStartMethod):
    """Meta logreg over [linmap, genre_affinity, genre_popularity], trained on val-cold.

    :param alpha: L2 regularization of the inner Ridge (linmap signal).
    :param C_reg: inverse regularization of the meta logistic regression.
    """

    requires = frozenset({"donor_scores", "content", "train_interactions", "val"})

    def __init__(self, alpha: float = 10.0, C_reg: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.C_reg = C_reg

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        warm = _req(inputs.warm_features, "warm_features")
        cold = _req(inputs.cold_features, "cold_features")
        val_cold = _req(inputs.val_cold_features, "val_cold_features")
        if inputs.val_interactions is None:
            raise ValueError("stacking_plus requires val_interactions")

        warm_mat = np.asarray(warm.matrix, dtype=float)
        warm_ids = np.asarray(warm.item_ids)

        # --- inner LinMap: Ridge content -> donor scores [n_warm, n_users] ---
        targets, self._user_ids = _pivot_scores_t(inputs.donor_scores, warm_ids)
        self._u_pos = {u: i for i, u in enumerate(self._user_ids)}
        self._ridge = Ridge(alpha=self.alpha)
        self._ridge.fit(np.asarray(_dense(warm_mat), dtype=float), targets)

        # --- personalized popularity ---
        self._affinity = _user_genre_affinity(
            inputs.train_interactions, warm_ids, warm_mat, self._u_pos
        )
        warm_pop = _warm_popularity(inputs.train_interactions, warm_ids)
        self._genre_pop = warm_mat.T @ warm_pop

        self._cold = cold

        # --- training the meta on val-cold ---
        x_val = self._features(val_cold, self._user_ids)
        y_val = _labels(inputs.val_interactions, self._user_ids, val_cold.item_ids)
        self._scaler = StandardScaler().fit(x_val)
        self._meta = LogisticRegression(C=self.C_reg, max_iter=1000, class_weight="balanced")
        self._meta.fit(self._scaler.transform(x_val), y_val)

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        user_ids = np.asarray(user_ids)
        cold_item_ids = np.asarray(cold_item_ids)

        cold_sub = self._cold.subset(cold_item_ids)
        x = self._features(cold_sub, user_ids)
        proba = self._meta.predict_proba(self._scaler.transform(x))[:, 1]
        scores = proba.reshape(len(user_ids), len(cold_item_ids))
        return cross_join_frame(user_ids, cold_item_ids, scores)

    def _features(self, item_feats: ItemFeatures, users: np.ndarray) -> np.ndarray:
        """Three signals for all pairs (users × item_feats), flattened into [n*m, 3]."""
        item_mat = np.asarray(item_feats.matrix, dtype=float)  # [m, n_genres]
        x_items = np.asarray(_dense(item_mat), dtype=float)

        # linmap signal: Ridge.predict gives [m, n_users_all]; select our users
        pred = np.asarray(self._ridge.predict(x_items))  # [m, n_users_all]
        cols = np.array([self._u_pos.get(u, -1) for u in users])
        known = cols >= 0
        linmap = np.zeros((len(users), item_mat.shape[0]))  # [n_users, m]
        linmap[known] = pred[:, cols[known]].T

        # personalized affinity and global popularity
        aff = np.zeros((len(users), self._affinity.shape[1]))
        aff[known] = self._affinity[cols[known]]
        aff_score = aff @ item_mat.T  # [n_users, m]
        pop_score = np.tile(item_mat @ self._genre_pop, (len(users), 1))  # [n_users, m]

        return np.column_stack(
            [linmap.reshape(-1), aff_score.reshape(-1), pop_score.reshape(-1)]
        )

    def get_params(self) -> dict:
        return {"alpha": self.alpha, "C_reg": self.C_reg}


def _req(feats: ItemFeatures | None, what: str) -> ItemFeatures:
    if feats is None:
        raise ValueError(f"stacking_plus requires {what}")
    return feats
