"""AUC (площадь под ROC) через ранговую формулу Манна–Уитни.

Своя реализация (без sklearn в горячем пути), корректно обрабатывает ties через
усреднение рангов. Сверяется с ``sklearn.roc_auc_score`` в тестах.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC по бинарным меткам ``y_true`` и скорам ``y_score``.

    Возвращает ``nan``, если в выборке только один класс (AUC не определён).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata(y_score)  # средние ранги при ties
    sum_pos = ranks[y_true == 1].sum()
    # формула Манна–Уитни: (Σranks_pos - n_pos(n_pos+1)/2) / (n_pos·n_neg)
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
