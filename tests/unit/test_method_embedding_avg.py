"""Тесты метода embedding_avg: контракт, усреднение эмбеддингов соседей, dot-скор."""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.methods import methods
from warmtransfer.methods.embedding_avg import EmbeddingAverage
from warmtransfer.types import ItemFeatures, TransferInputs


def _inputs() -> TransferInputs:
    """Synthetic-фикстура с предсказуемыми эмбеддингами.

    2 жанра (оси d=2). Warm-айтемы: 10 (жанр0), 11 (жанр1).
    Cold-айтемы: 20 ближе к warm 10 (жанр0), 21 ближе к warm 11 (жанр1).
    User 1 любит жанр0 -> user_emb=[1,0]; user 2 любит жанр1 -> user_emb=[0,1].
    """
    warm = ItemFeatures(np.array([10, 11]), np.array([[1.0, 0.0], [0.0, 1.0]]), ["g0", "g1"])
    cold = ItemFeatures(np.array([20, 21]), np.array([[1.0, 0.0], [0.0, 1.0]]), ["g0", "g1"])
    # similarity [n_cold=2, n_warm=2]: cold 20 -> warm 10, cold 21 -> warm 11
    similarity = np.array([[0.9, 0.1], [0.1, 0.9]])
    embeddings = {
        "item": np.array([[1.0, 0.0], [0.0, 1.0]]),  # warm 10 -> жанр0, warm 11 -> жанр1
        "item_ids": np.array([10, 11]),
        "user": np.array([[1.0, 0.0], [0.0, 1.0]]),  # user 1 -> жанр0, user 2 -> жанр1
        "user_ids": np.array([1, 2]),
    }
    donor = pd.DataFrame(
        {C.User: [1, 1, 2, 2], C.Item: [10, 11, 10, 11], C.Score: [0.9, 0.1, 0.2, 0.8]}
    )
    train = pd.DataFrame({C.User: [1, 2], C.Item: [10, 11], C.Weight: 1.0, C.Datetime: 0})
    return TransferInputs(
        donor_scores=donor,
        train_interactions=train,
        warm_features=warm,
        cold_features=cold,
        similarity=similarity,
        embeddings=embeddings,
    )


def test_registered() -> None:
    assert methods.get("embedding_avg") is EmbeddingAverage


def test_output_schema() -> None:
    inp = _inputs()
    reco = EmbeddingAverage(k=2).fit(inp, seed=0).predict(np.array([1, 2]), np.array([20, 21]))
    assert set(C.Scores) <= set(reco.columns)
    assert len(reco) == 4  # 2 users x 2 cold


def test_predicted_order() -> None:
    inp = _inputs()
    # k=1: cold 20 берёт эмбеддинг warm 10 ([1,0]), cold 21 берёт warm 11 ([0,1])
    m = EmbeddingAverage(k=1).fit(inp, seed=0)
    reco = m.predict(np.array([1, 2]), np.array([20, 21]))
    pivot = reco.pivot(index=C.User, columns=C.Item, values=C.Score)
    # user1 ([1,0]) скорит cold 20 ([1,0]) выше, чем cold 21 ([0,1])
    assert pivot.loc[1, 20] > pivot.loc[1, 21]
    # user2 ([0,1]) — наоборот
    assert pivot.loc[2, 21] > pivot.loc[2, 20]
    # точные значения dot-произведения
    assert np.isclose(pivot.loc[1, 20], 1.0)
    assert np.isclose(pivot.loc[1, 21], 0.0)


def test_unknown_user_zero_score() -> None:
    inp = _inputs()
    m = EmbeddingAverage(k=2).fit(inp, seed=0)
    reco = m.predict(np.array([999]), np.array([20, 21]))
    assert np.allclose(reco[C.Score].to_numpy(), 0.0)


def test_avg_of_two_neighbors() -> None:
    inp = _inputs()
    # k=2: эмбеддинг каждого cold = среднее warm 10 и 11 = [0.5, 0.5]
    m = EmbeddingAverage(k=2).fit(inp, seed=0)
    reco = m.predict(np.array([1]), np.array([20]))
    # user1 ([1,0]) . [0.5,0.5] = 0.5
    assert np.isclose(reco[C.Score].to_numpy()[0], 0.5)
