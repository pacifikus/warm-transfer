"""KNN-агрегация скоров донора с дебиасингом популярности соседей.

Как наивный KNN (см. ``knn.py``), но из скора каждого warm-соседа вычитается его
средний по пользователям скор (прокси популярности). Это снижает вклад глобально
популярных соседей: они тянут выдачу к "что нравится всем", а не к персональным
предпочтениям пользователя.

    score(u, i) = Σ_j w_j · (donor[u, j] - colmean_j),

где ``colmean_j`` — среднее donor[:, j] по пользователям, а ``w_j`` — нормированное
по сумме контентное сходство (как в knn.py, clip_negative=True).
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
    """KNN по контентным соседям с вычитанием популярности соседа.

    :param k: число ближайших warm-соседей.
    :param clip_negative: обнулять отрицательные сходства (косинус может быть <0).
    """

    requires = frozenset({"donor_scores", "similarity", "content"})

    def __init__(self, k: int = 20, clip_negative: bool = True) -> None:
        super().__init__()
        self.k = k
        self.clip_negative = clip_negative

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        if inputs.warm_features is None or inputs.cold_features is None:
            raise ValueError("debiased_knn требует warm_features и cold_features")
        if inputs.similarity is None:
            raise ValueError("debiased_knn требует similarity [n_cold, n_warm]")

        self._warm_ids = np.asarray(inputs.warm_features.item_ids)
        self._cold_ids = np.asarray(inputs.cold_features.item_ids)
        self._cold_pos = {it: r for r, it in enumerate(self._cold_ids)}

        # матрица скоров донора: [n_users, n_warm], выровнена по self._warm_ids
        self._donor_matrix, self._user_ids = _pivot_scores(inputs.donor_scores, self._warm_ids)
        self._user_pos = {u: i for i, u in enumerate(self._user_ids)}

        # прокси популярности соседа: средний по пользователям скор столбца [n_warm]
        self._col_mean = self._donor_matrix.mean(axis=0)
        # дебиасированная матрица: из скора вычитается популярность соседа
        self._debiased = self._donor_matrix - self._col_mean

        # для каждого cold-айтема: индексы top-k соседей и нормированные веса
        sim = np.asarray(inputs.similarity, dtype=float)
        if self.clip_negative:
            sim = np.clip(sim, 0.0, None)
        self._neighbors, self._weights = _topk_neighbors(sim, self.k)

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        user_ids = np.asarray(user_ids)
        cold_item_ids = np.asarray(cold_item_ids)

        # строки дебиасированной матрицы для запрошенных пользователей
        # неизвестный пользователь → нулевая строка (нет персональных отклонений)
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
            # [n_users] = debiased[:, соседи] @ веса
            scores[:, c] = debiased[:, nb] @ w
        return cross_join_frame(user_ids, cold_item_ids, scores)

    def get_params(self) -> dict:
        return {"k": self.k, "clip_negative": self.clip_negative}


def _pivot_scores(
    donor_scores: pd.DataFrame, warm_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Свести длинные скоры донора в матрицу [n_users, n_warm], выровненную по ``warm_ids``."""
    w_pos = {it: j for j, it in enumerate(warm_ids)}
    user_ids = unique_sorted(donor_scores[C.User])
    u_pos = {u: i for i, u in enumerate(user_ids)}

    # скоры донора заданы по warm-айтемам → все item_id есть в w_pos
    matrix = np.zeros((len(user_ids), len(warm_ids)))
    rows = map_codes(donor_scores[C.User], u_pos)
    cols = map_codes(donor_scores[C.Item], w_pos)
    matrix[rows, cols] = donor_scores[C.Score].to_numpy()
    return matrix, user_ids


def _topk_neighbors(sim: np.ndarray, k: int) -> tuple[list, list]:
    """Для каждой строки sim вернуть индексы top-k и нормированные веса (Σ=1)."""
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
