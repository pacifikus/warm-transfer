"""Attention-KNN: агрегация скоров донора по контентным соседям с softmax-весами.

Как :class:`KNNScoreAggregation` (knn.py), но веса k соседей вычисляются не как
нормированное сырое сходство, а как ``softmax(sim_to_neighbor / temperature)`` —
внимание по контентному сходству (SimCSR-lite). Малая ``temperature`` заостряет
распределение: вес самого похожего соседа стремится к 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coldscore._pdutils import map_codes, unique_sorted
from coldscore.columns import Columns as C
from coldscore.methods.base import ColdStartMethod, cross_join_frame, register_method
from coldscore.types import TransferInputs


@register_method("attention_knn")
class AttentionKNN(ColdStartMethod):
    """Softmax-взвешенное среднее скоров донора по k контентным соседям.

    :param k: число ближайших warm-соседей.
    :param temperature: температура softmax; меньше → острее (доминирует топ-сосед).
    """

    requires = frozenset({"donor_scores", "similarity", "content"})

    def __init__(self, k: int = 20, temperature: float = 0.1) -> None:
        super().__init__()
        self.k = k
        self.temperature = temperature

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        if inputs.warm_features is None or inputs.cold_features is None:
            raise ValueError("attention_knn требует warm_features и cold_features")
        if inputs.similarity is None:
            raise ValueError("attention_knn требует similarity [n_cold, n_warm]")

        self._warm_ids = np.asarray(inputs.warm_features.item_ids)
        self._cold_ids = np.asarray(inputs.cold_features.item_ids)
        self._cold_pos = {it: r for r, it in enumerate(self._cold_ids)}

        # матрица скоров донора: [n_users, n_warm], выровнена по self._warm_ids
        self._donor_matrix, self._user_ids = _pivot_scores(inputs.donor_scores, self._warm_ids)
        self._user_pos = {u: i for i, u in enumerate(self._user_ids)}

        # для каждого cold-айтема: индексы top-k соседей и softmax-веса по сходству
        sim = np.asarray(inputs.similarity, dtype=float)
        self._neighbors, self._weights = _topk_softmax(sim, self.k, self.temperature)

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        user_ids = np.asarray(user_ids)
        cold_item_ids = np.asarray(cold_item_ids)

        # строки донор-матрицы для запрошенных пользователей (неизвестные → нули)
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
            # [n_users] = donor[:, соседи] @ веса
            scores[:, c] = donor[:, nb] @ w
        return cross_join_frame(user_ids, cold_item_ids, scores)

    def get_params(self) -> dict:
        return {"k": self.k, "temperature": self.temperature}


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


def _topk_softmax(sim: np.ndarray, k: int, temperature: float) -> tuple[list, list]:
    """Для каждой строки sim вернуть индексы top-k и softmax-веса по сходству (Σ=1).

    Веса = ``softmax(sim[idx] / temperature)`` с вычетом max перед exp для
    численной устойчивости.
    """
    n_warm = sim.shape[1]
    kk = min(k, n_warm)
    neighbors: list = []
    weights: list = []
    for row in sim:
        idx = np.argpartition(-row, kk - 1)[:kk]
        idx = idx[np.argsort(-row[idx])]
        logits = row[idx] / temperature
        logits = logits - logits.max()  # устойчивость softmax
        exp = np.exp(logits)
        w = exp / exp.sum()
        neighbors.append(idx)
        weights.append(w)
    return neighbors, weights
