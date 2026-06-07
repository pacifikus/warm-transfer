"""Smoke test for logreg_calib: trains on val-cold, returns the correct schema."""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.methods.logreg_calib import LogRegCalibration
from warmtransfer.types import ItemFeatures, TransferInputs


def _feats(ids: list[int], genres: list[int]) -> ItemFeatures:
    mat = np.zeros((len(ids), 2), dtype=float)
    for i, g in enumerate(genres):
        mat[i, g] = 1.0
    return ItemFeatures(np.array(ids), mat, ["g0", "g1"])


def _inputs() -> TransferInputs:
    warm = _feats([10, 11, 12, 13], [0, 0, 1, 1])
    sim_g0 = [1.0, 1.0, 0.0, 0.0]
    sim_g1 = [0.0, 0.0, 1.0, 1.0]
    donor = pd.DataFrame(
        {
            C.User: [1, 1, 1, 1, 2, 2, 2, 2],
            C.Item: [10, 11, 12, 13, 10, 11, 12, 13],
            C.Score: [0.9, 0.8, 0.1, 0.2, 0.1, 0.2, 0.9, 0.8],
        }
    )
    return TransferInputs(
        donor_scores=donor,
        train_interactions=pd.DataFrame(
            {C.User: [1, 2], C.Item: [10, 12], C.Weight: 1.0, C.Datetime: 0}
        ),
        warm_features=warm,
        cold_features=_feats([30, 31], [0, 1]),
        similarity=np.array([sim_g0, sim_g1]),
        val_cold_features=_feats([20, 21], [0, 1]),
        val_similarity=np.array([sim_g0, sim_g1]),
        val_interactions=pd.DataFrame(
            {C.User: [1, 2], C.Item: [20, 21], C.Weight: 1.0, C.Datetime: 0}
        ),
        warm_items=np.array([10, 11, 12, 13]),
        cold_items=np.array([30, 31]),
    )


def test_logreg_calib_runs() -> None:
    m = LogRegCalibration(k=4).fit(_inputs(), seed=0)
    reco = m.predict(np.array([1, 2]), np.array([30, 31]))
    assert set(C.Scores) <= set(reco.columns)
    assert len(reco) == 4
    assert reco[C.Score].between(0, 1).all()
