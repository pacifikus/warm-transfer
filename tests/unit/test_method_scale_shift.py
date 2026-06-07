"""scale_shift test: content-based KNN over scores + calibration to warm statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.methods.scale_shift import ScaleShift
from warmtransfer.types import ItemFeatures, TransferInputs


def _feats(ids: list[int], genres: list[int]) -> ItemFeatures:
    mat = np.zeros((len(ids), 2), dtype=float)
    for i, g in enumerate(genres):
        mat[i, g] = 1.0
    return ItemFeatures(np.array(ids), mat, ["g0", "g1"])


def _inputs() -> TransferInputs:
    warm = _feats([10, 11, 12, 13], [0, 0, 1, 1])
    # donor scores: user1 likes g0 items, user2 likes g1 items
    rows = []
    for u, (s10, s11, s12, s13) in [(1, (5, 4, 1, 0)), (2, (0, 1, 4, 5))]:
        for it, sc in zip([10, 11, 12, 13], [s10, s11, s12, s13], strict=True):
            rows.append({C.User: u, C.Item: it, C.Score: float(sc)})
    donor = pd.DataFrame(rows)
    sim = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])  # cold 30→g0, 31→g1
    return TransferInputs(
        donor_scores=donor,
        train_interactions=pd.DataFrame(
            {C.User: [1, 2], C.Item: [10, 12], C.Weight: 1.0, C.Datetime: 0}
        ),
        warm_features=warm,
        cold_features=_feats([30, 31], [0, 1]),
        similarity=sim,
        warm_items=np.array([10, 11, 12, 13]),
        cold_items=np.array([30, 31]),
    )


def test_scale_shift_personalizes() -> None:
    m = ScaleShift(k=2).fit(_inputs(), seed=0)
    reco = m.predict(np.array([1, 2]), np.array([30, 31]))
    assert set(C.Scores) <= set(reco.columns)
    pivot = reco.pivot(index=C.User, columns=C.Item, values=C.Score)
    assert pivot.loc[1, 30] > pivot.loc[1, 31]
    assert pivot.loc[2, 31] > pivot.loc[2, 30]


def test_scale_shift_calibrates_to_warm_scale() -> None:
    m = ScaleShift(k=2)
    m.fit(_inputs(), seed=0)
    reco = m.predict(np.array([1, 2]), np.array([30, 31]))
    # mean cold-output score is close to the warm shift mu* (not zero/raw level)
    assert abs(reco[C.Score].mean() - m._mu_star) < 1.0


def test_scale_shift_unknown_user_is_mu_star() -> None:
    m = ScaleShift(k=2)
    m.fit(_inputs(), seed=0)
    reco = m.predict(np.array([999]), np.array([30, 31]))
    assert np.allclose(reco[C.Score].to_numpy(), m._mu_star)


def test_scale_shift_deterministic() -> None:
    a = ScaleShift(k=2).fit(_inputs(), seed=0).predict(np.array([1, 2]), np.array([30, 31]))
    b = ScaleShift(k=2).fit(_inputs(), seed=0).predict(np.array([1, 2]), np.array([30, 31]))
    assert np.allclose(a[C.Score].to_numpy(), b[C.Score].to_numpy())


def test_scale_shift_params() -> None:
    assert ScaleShift(k=10, clip_negative=False).get_params() == {
        "k": 10,
        "clip_negative": False,
    }
