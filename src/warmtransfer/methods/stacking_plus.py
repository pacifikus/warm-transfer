"""Stacking+: мета-модель над сильным сигналом LinMap + персонализированной популярностью.

Мотивация (из бенчмарка): LinMap выигрывает по ранжированию, но на популярностно-
доминируемых доменах (Goodbooks) персонализированный Grouped MP сильнее в top-k. Базовый
``stacking`` использует слабый knn-сигнал донора и не дотягивает до LinMap. Здесь мета-логрег
обучается над тремя сигналами:

  1. ``linmap`` — Ridge content→скоры донора (сильное ранжирование, перенос латентной структуры);
  2. ``genre affinity`` — аффинность юзера к жанрам айтема (силён в top-k на популярных доменах);
  3. ``genre popularity`` — глобальная популярность жанров (бейзлайн как фича).

Веса подбираются логистической регрессией на val-cold фолде (факт взаимодействия как target),
поэтому метод адаптивно берёт лучшее из обоих миров и должен быть робастен по доменам.
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
    """Мета-логрег над [linmap, genre_affinity, genre_popularity], обучение на val-cold.

    :param alpha: L2-регуляризация внутреннего Ridge (linmap-сигнал).
    :param C_reg: обратная регуляризация мета-логрегрессии.
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
            raise ValueError("stacking_plus требует val_interactions")

        warm_mat = np.asarray(warm.matrix, dtype=float)
        warm_ids = np.asarray(warm.item_ids)

        # --- внутренний LinMap: Ridge content -> скоры донора [n_warm, n_users] ---
        targets, self._user_ids = _pivot_scores_t(inputs.donor_scores, warm_ids)
        self._u_pos = {u: i for i, u in enumerate(self._user_ids)}
        self._ridge = Ridge(alpha=self.alpha)
        self._ridge.fit(np.asarray(_dense(warm_mat), dtype=float), targets)

        # --- персонализированная популярность ---
        self._affinity = _user_genre_affinity(
            inputs.train_interactions, warm_ids, warm_mat, self._u_pos
        )
        warm_pop = _warm_popularity(inputs.train_interactions, warm_ids)
        self._genre_pop = warm_mat.T @ warm_pop

        self._cold = cold

        # --- обучение меты на val-cold ---
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
        """Три сигнала для всех пар (users × item_feats), сплющенные в [n*m, 3]."""
        item_mat = np.asarray(item_feats.matrix, dtype=float)  # [m, n_genres]
        x_items = np.asarray(_dense(item_mat), dtype=float)

        # linmap-сигнал: Ridge.predict даёт [m, n_users_all]; выбрать наших юзеров
        pred = np.asarray(self._ridge.predict(x_items))  # [m, n_users_all]
        cols = np.array([self._u_pos.get(u, -1) for u in users])
        known = cols >= 0
        linmap = np.zeros((len(users), item_mat.shape[0]))  # [n_users, m]
        linmap[known] = pred[:, cols[known]].T

        # персонализированная аффинность и глобальная популярность
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
        raise ValueError(f"stacking_plus требует {what}")
    return feats
