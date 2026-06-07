"""AUC (area under ROC) via the Mann-Whitney rank formula.

Custom implementation (without sklearn on the hot path), correctly handles ties by
averaging ranks. Cross-checked against ``sklearn.roc_auc_score`` in the tests.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC from binary labels ``y_true`` and scores ``y_score``.

    Returns ``nan`` if the sample contains only one class (AUC is undefined).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata(y_score)  # average ranks on ties
    sum_pos = ranks[y_true == 1].sum()
    # Mann-Whitney formula: (Σranks_pos - n_pos(n_pos+1)/2) / (n_pos·n_neg)
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
