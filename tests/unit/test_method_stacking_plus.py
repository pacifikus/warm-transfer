"""Test stacking_plus: linmap+affinity hybrid trains on val-cold and personalizes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.methods.stacking_plus import StackingPlus
from warmtransfer.types import ItemFeatures, TransferInputs


def _feats(ids: list[int], genres: list[int]) -> ItemFeatures:
    mat = np.zeros((len(ids), 2), dtype=float)
    for i, g in enumerate(genres):
        mat[i, g] = 1.0
    return ItemFeatures(np.array(ids), mat, ["g0", "g1"])


def _inputs() -> TransferInputs:
    warm = _feats([10, 11, 12, 13], [0, 0, 1, 1])
    train = pd.DataFrame(
        {C.User: [1, 1, 2, 2], C.Item: [10, 11, 12, 13], C.Weight: 1.0, C.Datetime: 0}
    )
    donor = pd.DataFrame(
        {
            C.User: [1, 1, 1, 1, 2, 2, 2, 2],
            C.Item: [10, 11, 12, 13, 10, 11, 12, 13],
            C.Score: [0.9, 0.8, 0.1, 0.2, 0.1, 0.2, 0.9, 0.8],
        }
    )
    val_inter = pd.DataFrame({C.User: [1, 2], C.Item: [20, 21], C.Weight: 1.0, C.Datetime: 0})
    return TransferInputs(
        donor_scores=donor,
        train_interactions=train,
        warm_features=warm,
        cold_features=_feats([30, 31], [0, 1]),
        val_cold_features=_feats([20, 21], [0, 1]),
        val_interactions=val_inter,
        warm_items=np.array([10, 11, 12, 13]),
        cold_items=np.array([30, 31]),
    )


def test_stacking_plus_personalizes() -> None:
    m = StackingPlus().fit(_inputs(), seed=0)
    reco = m.predict(np.array([1, 2]), np.array([30, 31]))
    pivot = reco.pivot(index=C.User, columns=C.Item, values=C.Score)
    assert pivot.loc[1, 30] > pivot.loc[1, 31]
    assert pivot.loc[2, 31] > pivot.loc[2, 30]


def test_stacking_plus_output_schema() -> None:
    m = StackingPlus().fit(_inputs(), seed=0)
    reco = m.predict(np.array([1, 2]), np.array([30, 31]))
    assert set(C.Scores) <= set(reco.columns)
    assert len(reco) == 4
    assert reco[C.Score].between(0, 1).all()


def test_stacking_plus_get_params() -> None:
    assert StackingPlus(alpha=5.0, C_reg=2.0).get_params() == {"alpha": 5.0, "C_reg": 2.0}
