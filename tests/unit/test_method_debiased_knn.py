"""Тесты метода debiased_knn: контракт, схема вывода и эффект дебиасинга популярности."""

from __future__ import annotations

import numpy as np
import pandas as pd

from coldscore.columns import Columns as C
from coldscore.methods.debiased_knn import DebiasedKNN
from coldscore.types import ItemFeatures, TransferInputs


def _inputs() -> TransferInputs:
    """Синтетика: cold-айтем 20 имеет двух соседей — популярного (10) и нишевого (11).

    Сосед 10 глобально популярен: высокий средний по пользователям скор донора.
    Сосед 11 нишевый: средний скор низкий, но для user 1 он выделяется.
    """
    warm = ItemFeatures(
        np.array([10, 11]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        ["f0", "f1"],
    )
    # cold-айтем 20 одинаково близок к обоим соседям → веса по 0.5
    cold = ItemFeatures(np.array([20]), np.array([[0.5, 0.5]]), ["f0", "f1"])
    similarity = np.array([[1.0, 1.0]])  # [n_cold=1, n_warm=2]

    train = pd.DataFrame(
        {C.User: [1, 2, 3], C.Item: [10, 10, 10], C.Weight: 1.0, C.Datetime: 0}
    )
    # сосед 10 — высокий скор у всех (популярный); сосед 11 — высокий только у user 1
    donor = pd.DataFrame(
        {
            C.User: [1, 1, 2, 2, 3, 3],
            C.Item: [10, 11, 10, 11, 10, 11],
            C.Score: [0.9, 0.9, 0.9, 0.0, 0.9, 0.0],
        }
    )
    return TransferInputs(
        donor_scores=donor,
        train_interactions=train,
        warm_features=warm,
        cold_features=cold,
        similarity=similarity,
        warm_items=np.array([10, 11]),
        cold_items=np.array([20]),
    )


def test_output_schema() -> None:
    inp = _inputs()
    reco = DebiasedKNN().fit(inp, seed=0).predict(np.array([1, 2]), np.array([20]))
    assert set(C.Scores) <= set(reco.columns)
    assert len(reco) == 2  # 2 users × 1 cold


def test_deterministic() -> None:
    inp = _inputs()
    a = DebiasedKNN().fit(inp, seed=0).predict(np.array([1, 2, 3]), np.array([20]))
    b = DebiasedKNN().fit(inp, seed=0).predict(np.array([1, 2, 3]), np.array([20]))
    assert np.allclose(a[C.Score].to_numpy(), b[C.Score].to_numpy())


def test_popular_neighbor_contribution_reduced() -> None:
    """Дебиасинг снижает вклад глобально популярного соседа относительно наивного KNN."""
    inp = _inputs()
    from coldscore.methods.knn import KNNScoreAggregation

    users = np.array([1, 2, 3])
    cold = np.array([20])

    naive = KNNScoreAggregation().fit(inp, seed=0).predict(users, cold)
    deb = DebiasedKNN().fit(inp, seed=0).predict(users, cold)

    naive_s = naive.set_index(C.User)[C.Score]
    deb_s = deb.set_index(C.User)[C.Score]

    # Популярный сосед 10 (colmean=0.9) в дебиасе вычитается → его вклад зануляется.
    # Нишевый сосед 11 (colmean=0.3) сохраняет персональную часть.
    # Для всех пользователей дебиасированный скор ниже наивного (популярность вырезана).
    for u in users:
        assert deb_s.loc[u] < naive_s.loc[u]

    # Персонализация усиливается: у user 1 (любит нишевый 11) дебиасированный скор
    # становится выше, чем у user 2/3, для которых 11 неинтересен.
    assert deb_s.loc[1] > deb_s.loc[2]
    assert deb_s.loc[1] > deb_s.loc[3]
