"""Stacking: мета-модель поверх базовых сигналов, обучаемая на val-cold фолде.

Главный кандидат обойти сильный бейзлайн (персонализированный Grouped MP). Идея: для
каждой пары (user, cold_item) считаем несколько дешёвых сигналов и обучаем логистическую
регрессию предсказывать факт взаимодействия — НА VAL-COLD ФОЛДЕ (айтемы, которых нет в
train донора, но с известными взаимодействиями). Затем применяем к тестовым cold-айтемам.

Сигналы (все leak-free, считаются из train + статичный контент + скоры донора):
  1. personalized genre affinity — аффинность юзера к жанрам айтема (как grouped_most_popular_pers);
  2. global genre popularity — популярность жанров айтема (как grouped_most_popular);
  3. knn donor score — взвешенный по контентному сходству скор донора по соседям.

Так бейзлайн используется КАК ФИЧА (а не конкурент), плюс добавляется персонализированный
сигнал донора — поэтому stacking почти гарантированно ≥ Grouped MP.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from coldscore._pdutils import map_codes, unique_sorted
from coldscore.columns import Columns as C
from coldscore.methods.base import ColdStartMethod, cross_join_frame, register_method
from coldscore.types import ItemFeatures, TransferInputs


@register_method("stacking")
class StackingTransfer(ColdStartMethod):
    """Мета-логрегрессия на сигналах [affinity, genre_pop, knn], обучаемая на val-cold.

    :param k: число контентных соседей для knn-сигнала.
    :param C_reg: обратная регуляризация логистической регрессии.
    """

    requires = frozenset({"donor_scores", "content", "train_interactions", "similarity", "val"})

    def __init__(self, k: int = 20, C_reg: float = 1.0) -> None:
        super().__init__()
        self.k = k
        self.C_reg = C_reg

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        warm = _req(inputs.warm_features, "warm_features")
        cold = _req(inputs.cold_features, "cold_features")
        val_cold = _req(inputs.val_cold_features, "val_cold_features")
        if inputs.similarity is None or inputs.val_similarity is None:
            raise ValueError("stacking требует similarity и val_similarity")
        if inputs.val_interactions is None:
            raise ValueError("stacking требует val_interactions")

        warm_mat = np.asarray(warm.matrix, dtype=float)

        # пользователи = столбцы матрицы скоров донора
        self._users = unique_sorted(inputs.donor_scores[C.User])
        self._u_pos = {u: i for i, u in enumerate(self._users)}

        # матрица скоров донора D [n_users, n_warm], выровнена по warm.item_ids
        self._donor = _pivot_donor(inputs.donor_scores, warm.item_ids, self._users, self._u_pos)

        # аффинность юзера к жанрам [n_users, n_genres] по train (только наши пользователи)
        self._affinity = _user_genre_affinity(
            inputs.train_interactions, warm.item_ids, warm_mat, self._u_pos
        )
        # глобальная популярность жанров [n_genres]
        warm_pop = _warm_popularity(inputs.train_interactions, warm.item_ids)
        self._genre_pop = warm_mat.T @ warm_pop

        self._cold = cold
        self._cold_sim = np.asarray(inputs.similarity, dtype=float)
        self._cold_pos = {it: r for r, it in enumerate(cold.item_ids)}

        # --- обучающая выборка на val-cold ---
        val_sim = np.asarray(inputs.val_similarity, dtype=float)
        x_val = self._features(val_cold, val_sim, self._users)  # [n_users*n_val, 3]
        y_val = _labels(inputs.val_interactions, self._users, val_cold.item_ids)

        self._scaler = StandardScaler().fit(x_val)
        self._meta = LogisticRegression(C=self.C_reg, max_iter=1000, class_weight="balanced")
        self._meta.fit(self._scaler.transform(x_val), y_val)

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        user_ids = np.asarray(user_ids)
        cold_item_ids = np.asarray(cold_item_ids)

        cold_sub = self._cold.subset(cold_item_ids)
        rows = [self._cold_pos[it] for it in cold_item_ids]
        sim = self._cold_sim[rows]

        x = self._features(cold_sub, sim, user_ids)  # [n_users*n_cold, 3]
        proba = self._meta.predict_proba(self._scaler.transform(x))[:, 1]
        scores = proba.reshape(len(user_ids), len(cold_item_ids))
        return cross_join_frame(user_ids, cold_item_ids, scores)

    def _features(
        self, item_feats: ItemFeatures, sim: np.ndarray, users: np.ndarray
    ) -> np.ndarray:
        """Три сигнала для всех пар (users × item_feats), сплющенные в [n*m, 3]."""
        item_mat = np.asarray(item_feats.matrix, dtype=float)  # [m, n_genres]
        u_rows = np.array([self._u_pos.get(u, -1) for u in users])
        known = u_rows >= 0

        aff = np.zeros((len(users), self._affinity.shape[1]))
        aff[known] = self._affinity[u_rows[known]]
        donor = np.zeros((len(users), self._donor.shape[1]))
        donor[known] = self._donor[u_rows[known]]

        aff_score = aff @ item_mat.T  # [n_users, m]
        pop_score = np.tile(item_mat @ self._genre_pop, (len(users), 1))  # [n_users, m]
        knn_score = _knn_scores(donor, sim, self.k)  # [n_users, m]

        return np.column_stack(
            [aff_score.reshape(-1), pop_score.reshape(-1), knn_score.reshape(-1)]
        )

    def get_params(self) -> dict:
        return {"k": self.k, "C_reg": self.C_reg}


# --- внутренние утилиты ---


def _req(feats: ItemFeatures | None, what: str) -> ItemFeatures:
    if feats is None:
        raise ValueError(f"stacking требует {what}")
    return feats


def _pivot_donor(
    donor_scores: pd.DataFrame, warm_ids: np.ndarray, users: np.ndarray, u_pos: dict
) -> np.ndarray:
    """Скоры донора в матрицу [n_users, n_warm], выровненную по warm_ids и users."""
    w_pos = {it: j for j, it in enumerate(warm_ids)}
    extra = set(unique_sorted(donor_scores[C.Item]).tolist()) - set(w_pos)
    if extra:
        raise ValueError(
            f"donor_scores содержат не-warm айтемы ({len(extra)} шт.) — нарушение "
            "warm-only контракта: stacking должен учиться только на warm-скорах"
        )
    matrix = np.zeros((len(users), len(warm_ids)))
    rows = map_codes(donor_scores[C.User], u_pos)
    cols = map_codes(donor_scores[C.Item], w_pos)
    matrix[rows, cols] = donor_scores[C.Score].to_numpy()
    return matrix


def _warm_popularity(train: pd.DataFrame, warm_ids: np.ndarray) -> np.ndarray:
    counts = train.groupby(C.Item).size().to_dict()
    return np.array([float(counts.get(it, 0.0)) for it in warm_ids])


def _user_genre_affinity(
    train: pd.DataFrame, warm_ids: np.ndarray, warm_mat: np.ndarray, u_pos: dict
) -> np.ndarray:
    """[n_users, n_genres]: сколько раз юзер взаимодействовал с айтемами каждого жанра."""
    w_pos = {it: j for j, it in enumerate(warm_ids)}
    mask = train[C.User].isin(list(u_pos)) & train[C.Item].isin(list(w_pos))
    sub = cast("pd.DataFrame", train[mask])
    rows = map_codes(sub[C.User], u_pos)
    cols = map_codes(sub[C.Item], w_pos)
    counts = np.zeros((len(u_pos), warm_mat.shape[0]))
    np.add.at(counts, (rows, cols), 1.0)
    return counts @ warm_mat


def _knn_scores(donor: np.ndarray, sim: np.ndarray, k: int) -> np.ndarray:
    """Взвешенный по сходству скор донора по top-k контентным соседям. [n_users, n_items]."""
    n_users = donor.shape[0]
    n_items, n_warm = sim.shape
    kk = min(k, n_warm)
    out = np.zeros((n_users, n_items))
    sim = np.clip(sim, 0.0, None)
    for c in range(n_items):
        row = sim[c]
        idx = np.argpartition(-row, kk - 1)[:kk]
        w = row[idx]
        total = w.sum()
        w = w / total if total > 0 else np.full(kk, 1.0 / kk)
        out[:, c] = donor[:, idx] @ w
    return out


def _labels(val_interactions: pd.DataFrame, users: np.ndarray, items: np.ndarray) -> np.ndarray:
    """Метки 1/0 для пар (users × items) в порядке flatten (user-major)."""
    pos = set(zip(val_interactions[C.User], val_interactions[C.Item], strict=True))
    labels = np.zeros((len(users), len(items)), dtype=int)
    for ui, u in enumerate(users):
        for ci, it in enumerate(items):
            if (u, it) in pos:
                labels[ui, ci] = 1
    return labels.reshape(-1)
