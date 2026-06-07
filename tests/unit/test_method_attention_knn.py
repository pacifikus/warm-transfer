"""Тесты метода attention_knn: контракт, softmax-веса, доминирование при малой T."""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.methods import methods
from warmtransfer.methods.attention_knn import AttentionKNN
from warmtransfer.types import ItemFeatures, TransferInputs


def _inputs() -> TransferInputs:
    # 3 warm-айтема (10, 11, 12) и 1 cold (20).
    warm = ItemFeatures(
        np.array([10, 11, 12]),
        np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        ["g0", "g1"],
    )
    cold = ItemFeatures(np.array([20]), np.array([[1.0, 0.0]]), ["g0", "g1"])
    train = pd.DataFrame({C.User: [1, 1, 1], C.Item: [10, 11, 12], C.Weight: 1.0, C.Datetime: 0})
    # один пользователь, скоры донора по warm-айтемам
    donor = pd.DataFrame({C.User: [1, 1, 1], C.Item: [10, 11, 12], C.Score: [1.0, 0.5, 0.0]})
    # cold-айтем 20 ближе всего к warm 10, затем 12, затем 11
    sim = np.array([[0.9, 0.1, 0.5]])
    return TransferInputs(
        donor_scores=donor,
        train_interactions=train,
        warm_features=warm,
        cold_features=cold,
        similarity=sim,
        warm_items=np.array([10, 11, 12]),
        cold_items=np.array([20]),
    )


def test_registered() -> None:
    assert methods.get("attention_knn") is AttentionKNN


def test_output_schema() -> None:
    m = AttentionKNN(k=3).fit(_inputs(), seed=0)
    reco = m.predict(np.array([1]), np.array([20]))
    assert set(C.Scores) <= set(reco.columns)
    assert len(reco) == 1


def test_get_params() -> None:
    m = AttentionKNN(k=5, temperature=0.3)
    assert m.get_params() == {"k": 5, "temperature": 0.3}


def test_small_temperature_dominates_top_neighbor() -> None:
    # При малой temperature вес самого похожего соседа (warm 10, sim=0.9) ≈ 1,
    # поэтому скор cold-айтема ≈ скор донора по warm 10 (=1.0).
    m = AttentionKNN(k=3, temperature=0.001).fit(_inputs(), seed=0)
    reco = m.predict(np.array([1]), np.array([20]))
    score = float(reco[C.Score].iloc[0])
    assert abs(score - 1.0) < 1e-3

    # явно проверим веса: топ-сосед доминирует
    top_weight = m._weights[0][0]  # type: ignore[attr-defined]
    assert top_weight > 0.999


def test_large_temperature_approaches_uniform() -> None:
    # При большой temperature softmax → равномерное распределение по k соседям.
    m = AttentionKNN(k=3, temperature=1e6).fit(_inputs(), seed=0)
    w = m._weights[0]  # type: ignore[attr-defined]
    assert np.allclose(w, np.full(3, 1.0 / 3), atol=1e-4)


def test_weights_sum_to_one() -> None:
    m = AttentionKNN(k=3, temperature=0.1).fit(_inputs(), seed=0)
    w = m._weights[0]  # type: ignore[attr-defined]
    assert abs(float(w.sum()) - 1.0) < 1e-9
