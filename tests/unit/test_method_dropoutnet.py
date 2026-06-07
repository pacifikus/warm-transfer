"""DropoutNet test: trains on warm embeddings, predicts cold via content."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from warmtransfer.columns import Columns as C
from warmtransfer.types import ItemFeatures, TransferInputs

pytestmark = pytest.mark.bench  # requires torch (extra deep)


def _feats(ids: list[int], genres: list[int]) -> ItemFeatures:
    mat = np.zeros((len(ids), 2), dtype=float)
    for i, g in enumerate(genres):
        mat[i, g] = 1.0
    return ItemFeatures(np.array(ids), mat, ["g0", "g1"])


def _inputs() -> TransferInputs:
    warm = _feats([10, 11, 12, 13], [0, 0, 1, 1])
    # donor latent: g0 items are similar to user1, g1 items — to user2
    item_emb = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=float
    )
    user_emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    return TransferInputs(
        donor_scores=pd.DataFrame({C.User: [1], C.Item: [10], C.Score: [1.0]}),
        train_interactions=pd.DataFrame(
            {C.User: [1, 2], C.Item: [10, 12], C.Weight: 1.0, C.Datetime: 0}
        ),
        warm_features=warm,
        cold_features=_feats([30, 31], [0, 1]),
        embeddings={
            "item": item_emb,
            "item_ids": np.array([10, 11, 12, 13]),
            "user": user_emb,
            "user_ids": np.array([1, 2]),
        },
        warm_items=np.array([10, 11, 12, 13]),
        cold_items=np.array([30, 31]),
    )


def test_dropoutnet_learns_content_to_latent() -> None:
    from warmtransfer.methods.dropoutnet import DropoutNet

    m = DropoutNet(hidden=16, epochs=300, dropout_pref=0.5).fit(_inputs(), seed=0)
    reco = m.predict(np.array([1, 2]), np.array([30, 31]))
    assert set(C.Scores) <= set(reco.columns)
    assert len(reco) == 4
    pivot = reco.pivot(index=C.User, columns=C.Item, values=C.Score)
    # user1 (latent g0) scores cold g0 (30) higher; user2 (g1) — cold g1 (31)
    assert pivot.loc[1, 30] > pivot.loc[1, 31]
    assert pivot.loc[2, 31] > pivot.loc[2, 30]


def test_dropoutnet_params() -> None:
    from warmtransfer.methods.dropoutnet import DropoutNet

    p = DropoutNet(hidden=32, epochs=10).get_params()
    assert p["hidden"] == 32 and p["epochs"] == 10
