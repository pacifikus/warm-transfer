"""AUC tests: known-answer, ties, single class, cross-check against sklearn."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from warmtransfer.metrics.classification import auc
from warmtransfer.metrics.relative import rela_impr


def test_auc_known_answer() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    assert auc(y_true, y_score) == pytest.approx(0.75)


def test_auc_ties_half() -> None:
    assert auc(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_auc_single_class_nan() -> None:
    assert math.isnan(auc(np.array([1, 1]), np.array([0.2, 0.9])))
    assert math.isnan(auc(np.array([0, 0]), np.array([0.2, 0.9])))


def test_auc_matches_sklearn() -> None:
    rng = np.random.default_rng(42)
    for _ in range(20):
        y_true = rng.integers(0, 2, size=50)
        if y_true.min() == y_true.max():
            continue
        y_score = rng.random(50)
        assert auc(y_true, y_score) == pytest.approx(roc_auc_score(y_true, y_score))


def test_rela_impr() -> None:
    # (0.7-0.5)/(0.6-0.5) - 1 = 2 - 1 = 1.0
    assert rela_impr(0.7, 0.6) == pytest.approx(1.0)
    assert math.isnan(rela_impr(0.7, 0.5))  # baseline at the random level
