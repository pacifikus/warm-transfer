"""Tests for the debiased_knn method: contract, output schema, popularity debiasing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.methods.debiased_knn import DebiasedKNN
from warmtransfer.types import ItemFeatures, TransferInputs


def _inputs() -> TransferInputs:
    """Synthetic data: cold item 20 has two neighbors — popular (10) and niche (11).

    Neighbor 10 is globally popular: a high donor score averaged over users.
    Neighbor 11 is niche: low average score, but it stands out for user 1.
    """
    warm = ItemFeatures(
        np.array([10, 11]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        ["f0", "f1"],
    )
    # cold item 20 is equally close to both neighbors → weights of 0.5 each
    cold = ItemFeatures(np.array([20]), np.array([[0.5, 0.5]]), ["f0", "f1"])
    similarity = np.array([[1.0, 1.0]])  # [n_cold=1, n_warm=2]

    train = pd.DataFrame(
        {C.User: [1, 2, 3], C.Item: [10, 10, 10], C.Weight: 1.0, C.Datetime: 0}
    )
    # neighbor 10 — high score for everyone (popular); neighbor 11 — high only for user 1
    donor = pd.DataFrame(
        {
            C.User: [1, 1, 2, 2, 3, 3],
            C.Item: [10, 11, 10, 11, 10, 11],
            C.Score: [0.9, 0.9, 0.9, 0.0, 0.9, 0.0],
        }
    )
    return TransferInputs(
        donor_scores=donor,
        train_interactions=train,
        warm_features=warm,
        cold_features=cold,
        similarity=similarity,
        warm_items=np.array([10, 11]),
        cold_items=np.array([20]),
    )


def test_output_schema() -> None:
    inp = _inputs()
    reco = DebiasedKNN().fit(inp, seed=0).predict(np.array([1, 2]), np.array([20]))
    assert set(C.Scores) <= set(reco.columns)
    assert len(reco) == 2  # 2 users × 1 cold


def test_deterministic() -> None:
    inp = _inputs()
    a = DebiasedKNN().fit(inp, seed=0).predict(np.array([1, 2, 3]), np.array([20]))
    b = DebiasedKNN().fit(inp, seed=0).predict(np.array([1, 2, 3]), np.array([20]))
    assert np.allclose(a[C.Score].to_numpy(), b[C.Score].to_numpy())


def test_popular_neighbor_contribution_reduced() -> None:
    """Debiasing reduces the globally popular neighbor's contribution vs. naive KNN."""
    inp = _inputs()
    from warmtransfer.methods.knn import KNNScoreAggregation

    users = np.array([1, 2, 3])
    cold = np.array([20])

    naive = KNNScoreAggregation().fit(inp, seed=0).predict(users, cold)
    deb = DebiasedKNN().fit(inp, seed=0).predict(users, cold)

    naive_s = naive.set_index(C.User)[C.Score]
    deb_s = deb.set_index(C.User)[C.Score]

    # Popular neighbor 10 (colmean=0.9) is subtracted in debiasing → its contribution
    # zeroes out.
    # Niche neighbor 11 (colmean=0.3) keeps its personal part.
    # For every user the debiased score is below the naive one (popularity stripped out).
    for u in users:
        assert deb_s.loc[u] < naive_s.loc[u]

    # Personalization is amplified: for user 1 (who likes the niche item 11) the debiased
    # score becomes higher than for users 2/3, for whom 11 is irrelevant.
    assert deb_s.loc[1] > deb_s.loc[2]
    assert deb_s.loc[1] > deb_s.loc[3]
