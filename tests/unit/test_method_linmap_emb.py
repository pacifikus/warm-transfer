"""Тест linmap_emb (Gantner): Ridge контент → латентные факторы донора."""

from __future__ import annotations

import numpy as np
import pandas as pd

from coldscore.columns import Columns as C
from coldscore.methods.linmap_emb import LinMapEmbedding
from coldscore.types import ItemFeatures, TransferInputs


def _feats(ids: list[int], genres: list[int]) -> ItemFeatures:
    mat = np.zeros((len(ids), 2), dtype=float)
    for i, g in enumerate(genres):
        mat[i, g] = 1.0
    return ItemFeatures(np.array(ids), mat, ["g0", "g1"])


def _inputs() -> TransferInputs:
    # warm: жанр g0 → фактор [1,0], жанр g1 → фактор [0,1]
    warm = _feats([10, 11, 12, 13], [0, 0, 1, 1])
    item_emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=float)
    user_emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    return TransferInputs(
        donor_scores=pd.DataFrame({C.User: [1], C.Item: [10], C.Score: [1.0]}),
        train_interactions=pd.DataFrame(
            {C.User: [1, 2], C.Item: [10, 12], C.Weight: 1.0, C.Datetime: 0}
        ),
        warm_features=warm,
        cold_features=_feats([30, 31], [0, 1]),  # cold 30→g0, 31→g1
        embeddings={
            "item": item_emb,
            "item_ids": np.array([10, 11, 12, 13]),
            "user": user_emb,
            "user_ids": np.array([1, 2]),
        },
        warm_items=np.array([10, 11, 12, 13]),
        cold_items=np.array([30, 31]),
    )


def test_linmap_emb_maps_content_to_factors() -> None:
    m = LinMapEmbedding(alpha=0.01).fit(_inputs(), seed=0)
    reco = m.predict(np.array([1, 2]), np.array([30, 31]))
    assert set(C.Scores) <= set(reco.columns)
    assert len(reco) == 4
    pivot = reco.pivot(index=C.User, columns=C.Item, values=C.Score)
    # user1 (фактор g0) выше оценивает cold 30 (g0); user2 — cold 31 (g1)
    assert pivot.loc[1, 30] > pivot.loc[1, 31]
    assert pivot.loc[2, 31] > pivot.loc[2, 30]


def test_linmap_emb_unknown_user_zero() -> None:
    m = LinMapEmbedding(alpha=0.01).fit(_inputs(), seed=0)
    reco = m.predict(np.array([999]), np.array([30, 31]))
    assert np.allclose(reco[C.Score].to_numpy(), 0.0)


def test_linmap_emb_deterministic() -> None:
    a = LinMapEmbedding(alpha=1.0).fit(_inputs(), seed=0)
    b = LinMapEmbedding(alpha=1.0).fit(_inputs(), seed=0)
    ra = a.predict(np.array([1, 2]), np.array([30, 31]))[C.Score].to_numpy()
    rb = b.predict(np.array([1, 2]), np.array([30, 31]))[C.Score].to_numpy()
    assert np.allclose(ra, rb)


def test_linmap_emb_params() -> None:
    assert LinMapEmbedding(alpha=5.0).get_params() == {"alpha": 5.0}
