"""Тест метода linmap: Ridge-отображение контента в скоры донора."""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.methods.linmap import LinMap
from warmtransfer.types import ItemFeatures, TransferInputs


def _inputs() -> TransferInputs:
    # 2 жанра. warm: 10,12 (жанр0), 11,13 (жанр1); cold: 20 (жанр0), 21 (жанр1)
    warm = ItemFeatures(
        np.array([10, 11, 12, 13]),
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        ["g0", "g1"],
    )
    cold = ItemFeatures(
        np.array([20, 21]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        ["g0", "g1"],
    )
    train = pd.DataFrame({C.User: [1, 2], C.Item: [10, 11], C.Weight: 1.0, C.Datetime: 0})
    # user1 высоко скорит жанр0 (айтемы 10,12), низко жанр1; user2 — наоборот
    donor = pd.DataFrame(
        {
            C.User: [1, 1, 1, 1, 2, 2, 2, 2],
            C.Item: [10, 11, 12, 13, 10, 11, 12, 13],
            C.Score: [0.9, 0.1, 0.8, 0.2, 0.2, 0.8, 0.1, 0.9],
        }
    )
    return TransferInputs(
        donor_scores=donor,
        train_interactions=train,
        warm_features=warm,
        cold_features=cold,
        warm_items=np.array([10, 11, 12, 13]),
        cold_items=np.array([20, 21]),
    )


def test_linmap_transfers_genre_preference() -> None:
    m = LinMap(alpha=1.0).fit(_inputs(), seed=0)
    reco = m.predict(np.array([1, 2]), np.array([20, 21]))
    pivot = reco.pivot(index=C.User, columns=C.Item, values=C.Score)
    # cold-айтем 20 (жанр0) у user1 скорится выше, чем cold-айтем 21 (другой жанр)
    assert pivot.loc[1, 20] > pivot.loc[1, 21]
    # симметрично для user2 (жанр1)
    assert pivot.loc[2, 21] > pivot.loc[2, 20]


def test_linmap_output_schema_and_unknown_user() -> None:
    m = LinMap().fit(_inputs(), seed=0)
    reco = m.predict(np.array([1, 999]), np.array([20, 21]))
    assert set(C.Scores) <= set(reco.columns)
    assert len(reco) == 4  # 2 users × 2 cold
    # неизвестный пользователь получает нулевые скоры
    unknown = reco.loc[reco[C.User] == 999, C.Score]
    assert np.allclose(unknown.to_numpy(), 0.0)


def test_linmap_get_params() -> None:
    assert LinMap(alpha=5.0).get_params() == {"alpha": 5.0}
