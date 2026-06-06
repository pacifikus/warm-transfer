"""Тесты бейзлайнов: контракт, детерминизм, персонализированный GroupedMP."""

from __future__ import annotations

import numpy as np
import pandas as pd

from coldscore.columns import Columns as C
from coldscore.methods import methods
from coldscore.types import ItemFeatures, TransferInputs


def _inputs() -> TransferInputs:
    # 2 жанра; warm-айтемы: 10 (жанр0), 11 (жанр1); cold: 20 (жанр0), 21 (жанр1)
    warm = ItemFeatures(np.array([10, 11]), np.array([[1.0, 0.0], [0.0, 1.0]]), ["g0", "g1"])
    cold = ItemFeatures(np.array([20, 21]), np.array([[1.0, 0.0], [0.0, 1.0]]), ["g0", "g1"])
    # user 1 любит жанр0 (взаимодействовал с 10), user 2 — жанр1 (с 11)
    train = pd.DataFrame(
        {C.User: [1, 2], C.Item: [10, 11], C.Weight: 1.0, C.Datetime: 0}
    )
    donor = pd.DataFrame(
        {C.User: [1, 1, 2, 2], C.Item: [10, 11, 10, 11], C.Score: [0.9, 0.1, 0.2, 0.8]}
    )
    return TransferInputs(
        donor_scores=donor,
        train_interactions=train,
        warm_features=warm,
        cold_features=cold,
        warm_items=np.array([10, 11]),
        cold_items=np.array([20, 21]),
    )


def test_random_deterministic() -> None:
    inp = _inputs()
    a = methods.get("random")().fit(inp, seed=1).predict(np.array([1, 2]), np.array([20, 21]))
    b = methods.get("random")().fit(inp, seed=1).predict(np.array([1, 2]), np.array([20, 21]))
    assert np.allclose(a[C.Score].to_numpy(), b[C.Score].to_numpy())


def test_grouped_pers_personalizes() -> None:
    inp = _inputs()
    m = methods.get("grouped_most_popular_pers")().fit(inp, seed=0)
    reco = m.predict(np.array([1, 2]), np.array([20, 21]))
    pivot = reco.pivot(index=C.User, columns=C.Item, values=C.Score)
    # user1 (жанр0) выше скорит cold-айтем 20 (жанр0), чем 21
    assert pivot.loc[1, 20] > pivot.loc[1, 21]
    # user2 (жанр1) — наоборот
    assert pivot.loc[2, 21] > pivot.loc[2, 20]


def test_grouped_global_same_across_users() -> None:
    inp = _inputs()
    m = methods.get("grouped_most_popular")().fit(inp, seed=0)
    reco = m.predict(np.array([1, 2]), np.array([20, 21]))
    pivot = reco.pivot(index=C.User, columns=C.Item, values=C.Score)
    # глобальный GroupedMP: скор айтема одинаков для всех пользователей
    assert pivot.loc[1, 20] == pivot.loc[2, 20]


def test_all_methods_output_schema() -> None:
    inp = _inputs()
    for name in ["random", "most_popular", "grouped_most_popular", "grouped_most_popular_pers"]:
        reco = methods.get(name)().fit(inp, seed=0).predict(np.array([1, 2]), np.array([20, 21]))
        assert set(C.Scores) <= set(reco.columns)
        assert len(reco) == 4  # 2 users × 2 cold
