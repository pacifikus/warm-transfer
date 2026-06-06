"""Контентное сходство cold→warm айтемов (косинус).

Может делегироваться пользователю библиотеки (он передаёт готовую матрицу сходства),
либо строиться здесь из ``ItemFeatures``.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from coldscore.types import ItemFeatures


def content_similarity(cold: ItemFeatures, warm: ItemFeatures) -> np.ndarray:
    """Косинусное сходство [n_cold, n_warm] между cold и warm айтемами.

    Строки выровнены по ``cold.item_ids``, столбцы — по ``warm.item_ids``.
    """
    cold_mat = np.asarray(cold.matrix, dtype=float)
    warm_mat = np.asarray(warm.matrix, dtype=float)
    return cosine_similarity(cold_mat, warm_mat)
