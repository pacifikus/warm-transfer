"""Test attention_emb: softmax-weighted averaging of content-neighbor embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.methods.attention_emb import AttentionEmbedding
from warmtransfer.types import ItemFeatures, TransferInputs


def _feats(ids: list[int], genres: list[int]) -> ItemFeatures:
    mat = np.zeros((len(ids), 2), dtype=float)
    for i, g in enumerate(genres):
        mat[i, g] = 1.0
    return ItemFeatures(np.array(ids), mat, ["g0", "g1"])


def _inputs() -> TransferInputs:
    warm = _feats([10, 11, 12, 13], [0, 0, 1, 1])
    item_emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=float)
    user_emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    sim = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])  # cold[30->g0,31->g1]
    return TransferInputs(
        donor_scores=pd.DataFrame({C.User: [1], C.Item: [10], C.Score: [1.0]}),
        train_interactions=pd.DataFrame(
            {C.User: [1, 2], C.Item: [10, 12], C.Weight: 1.0, C.Datetime: 0}
        ),
        warm_features=warm,
        cold_features=_feats([30, 31], [0, 1]),
        similarity=sim,
        embeddings={
            "item": item_emb,
            "item_ids": np.array([10, 11, 12, 13]),
            "user": user_emb,
            "user_ids": np.array([1, 2]),
        },
        warm_items=np.array([10, 11, 12, 13]),
        cold_items=np.array([30, 31]),
    )


def test_attention_emb_personalizes() -> None:
    m = AttentionEmbedding(k=4, temperature=0.1).fit(_inputs(), seed=0)
    reco = m.predict(np.array([1, 2]), np.array([30, 31]))
    assert set(C.Scores) <= set(reco.columns)
    assert len(reco) == 4
    pivot = reco.pivot(index=C.User, columns=C.Item, values=C.Score)
    assert pivot.loc[1, 30] > pivot.loc[1, 31]
    assert pivot.loc[2, 31] > pivot.loc[2, 30]


def test_attention_emb_unknown_user_zero() -> None:
    m = AttentionEmbedding(k=4).fit(_inputs(), seed=0)
    reco = m.predict(np.array([999]), np.array([30, 31]))
    assert np.allclose(reco[C.Score].to_numpy(), 0.0)


def test_attention_emb_params() -> None:
    assert AttentionEmbedding(k=10, temperature=0.5).get_params() == {
        "k": 10,
        "temperature": 0.5,
    }
