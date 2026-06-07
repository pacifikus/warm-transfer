"""Тест magnitude_scaling: дебиасинг популярности по магнитуде cold-эмбеддинга."""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.methods.linmap_emb import LinMapEmbedding
from warmtransfer.methods.magnitude_scaling import MagnitudeScaling
from warmtransfer.types import ItemFeatures, TransferInputs


def _feats(ids: list[int], rows: list[list[float]]) -> ItemFeatures:
    return ItemFeatures(np.array(ids), np.array(rows, dtype=float), ["f0", "f1"])


def _inputs() -> TransferInputs:
    # warm-эмбеддинги с нормами 1 и 3 → mu_w = 2
    warm = _feats([10, 11], [[1.0, 0.0], [0.0, 1.0]])
    item_emb = np.array([[1.0, 0.0], [3.0, 0.0]], dtype=float)
    user_emb = np.array([[1.0, 0.0]], dtype=float)
    # cold-контент совпадает с warm → факторы будут ~ соответствующие нормы
    return TransferInputs(
        donor_scores=pd.DataFrame({C.User: [1], C.Item: [10], C.Score: [1.0]}),
        train_interactions=pd.DataFrame(
            {C.User: [1], C.Item: [10], C.Weight: 1.0, C.Datetime: 0}
        ),
        warm_features=warm,
        cold_features=_feats([30, 31], [[1.0, 0.0], [0.0, 1.0]]),
        embeddings={
            "item": item_emb,
            "item_ids": np.array([10, 11]),
            "user": user_emb,
            "user_ids": np.array([1]),
        },
        warm_items=np.array([10, 11]),
        cold_items=np.array([30, 31]),
    )


def test_ms_pulls_norms_toward_warm_mean() -> None:
    base = LinMapEmbedding(alpha=1.0)
    base.fit(_inputs(), seed=0)
    ms = MagnitudeScaling(alpha=1.0, ms_alpha=1.0)
    ms.fit(_inputs(), seed=0)
    base_norms = np.linalg.norm(base._cold_emb, axis=1)
    ms_norms = np.linalg.norm(ms._cold_emb, axis=1)
    mu_w = ms._warm_mean_norm
    # дисперсия норм после стягивания меньше — разброс популярности сжат
    assert np.std(ms_norms) < np.std(base_norms)
    # каждая норма ближе к mu_w, чем была
    assert np.all(np.abs(ms_norms - mu_w) <= np.abs(base_norms - mu_w) + 1e-9)


def test_ms_alpha_zero_is_identity() -> None:
    base = LinMapEmbedding(alpha=1.0)
    base.fit(_inputs(), seed=0)
    ms = MagnitudeScaling(alpha=1.0, ms_alpha=0.0)
    ms.fit(_inputs(), seed=0)
    assert np.allclose(base._cold_emb, ms._cold_emb)


def test_ms_direction_preserved() -> None:
    base = LinMapEmbedding(alpha=1.0)
    base.fit(_inputs(), seed=0)
    ms = MagnitudeScaling(alpha=1.0, ms_alpha=2.0)
    ms.fit(_inputs(), seed=0)
    # направление не меняется — только длина
    for b, s in zip(base._cold_emb, ms._cold_emb, strict=True):
        nb, ns = np.linalg.norm(b), np.linalg.norm(s)
        if nb > 0 and ns > 0:
            assert np.allclose(b / nb, s / ns)


def test_ms_predict_schema() -> None:
    ms = MagnitudeScaling()
    ms.fit(_inputs(), seed=0)
    reco = ms.predict(np.array([1]), np.array([30, 31]))
    assert set(C.Scores) <= set(reco.columns)
    assert len(reco) == 2


def test_ms_params() -> None:
    assert MagnitudeScaling(alpha=5.0, ms_alpha=0.5).get_params() == {
        "alpha": 5.0,
        "ms_alpha": 0.5,
    }
