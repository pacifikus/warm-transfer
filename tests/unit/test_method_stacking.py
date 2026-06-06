"""Тест stacking: обучается на val-cold и переносит персонализацию на test-cold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from coldscore.columns import Columns as C
from coldscore.methods.stacking import StackingTransfer
from coldscore.types import ItemFeatures, TransferInputs


def _feats(ids: list[int], genres: list[int]) -> ItemFeatures:
    mat = np.zeros((len(ids), 2), dtype=float)
    for i, g in enumerate(genres):
        mat[i, g] = 1.0
    return ItemFeatures(np.array(ids), mat, ["g0", "g1"])


def _inputs() -> TransferInputs:
    warm = _feats([10, 11, 12, 13], [0, 0, 1, 1])
    cold = _feats([30, 31], [0, 1])
    val_cold = _feats([20, 21], [0, 1])

    train = pd.DataFrame(
        {
            C.User: [1, 1, 2, 2],
            C.Item: [10, 11, 12, 13],
            C.Weight: 1.0,
            C.Datetime: 0,
        }
    )
    donor = pd.DataFrame(
        {
            C.User: [1, 1, 1, 1, 2, 2, 2, 2],
            C.Item: [10, 11, 12, 13, 10, 11, 12, 13],
            C.Score: [0.9, 0.8, 0.1, 0.2, 0.1, 0.2, 0.9, 0.8],
        }
    )
    # similarity по one-hot жанрам: g0-айтем похож на warm 10,11; g1 — на 12,13
    sim_g0 = [1.0, 1.0, 0.0, 0.0]
    sim_g1 = [0.0, 0.0, 1.0, 1.0]
    similarity = np.array([sim_g0, sim_g1])      # cold [30(g0), 31(g1)]
    val_similarity = np.array([sim_g0, sim_g1])  # val  [20(g0), 21(g1)]
    # user1 взаимодействовал с val g0 (20), user2 — с val g1 (21)
    val_inter = pd.DataFrame({C.User: [1, 2], C.Item: [20, 21], C.Weight: 1.0, C.Datetime: 0})

    return TransferInputs(
        donor_scores=donor,
        train_interactions=train,
        warm_features=warm,
        cold_features=cold,
        similarity=similarity,
        val_cold_features=val_cold,
        val_similarity=val_similarity,
        val_interactions=val_inter,
        warm_items=np.array([10, 11, 12, 13]),
        cold_items=np.array([30, 31]),
    )


def test_stacking_personalizes() -> None:
    m = StackingTransfer(k=4).fit(_inputs(), seed=0)
    reco = m.predict(np.array([1, 2]), np.array([30, 31]))
    pivot = reco.pivot(index=C.User, columns=C.Item, values=C.Score)
    # user1 (любит g0) выше скорит cold g0 (30); user2 (g1) — cold g1 (31)
    assert pivot.loc[1, 30] > pivot.loc[1, 31]
    assert pivot.loc[2, 31] > pivot.loc[2, 30]


def test_stacking_output_schema() -> None:
    m = StackingTransfer(k=4).fit(_inputs(), seed=0)
    reco = m.predict(np.array([1, 2]), np.array([30, 31]))
    assert set(C.Scores) <= set(reco.columns)
    assert len(reco) == 4
