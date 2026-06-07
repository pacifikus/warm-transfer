"""Test for the TwoTowerAdapter donor: training + scoring on a small synthetic dataset.

Two-Tower is a neural donor (torch) with a dot-product score, so it DOES expose a latent
space — ``embeddings()`` must return user/item vectors. Marked ``bench`` because it
requires torch (extra ``deep``); skipped when torch is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from warmtransfer.bench.adapters.two_tower import TwoTowerAdapter
from warmtransfer.columns import Columns as C
from warmtransfer.types import Dataset

pytestmark = pytest.mark.bench


def _dataset() -> Dataset:
    """5 users, 6 items."""
    users = [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5]
    items = [10, 11, 12, 10, 13, 11, 14, 12, 15, 13, 14, 15]
    inter = pd.DataFrame({C.User: users, C.Item: items, C.Weight: 1.0, C.Datetime: 0})
    return Dataset(interactions=inter, name="tiny")


def _model() -> TwoTowerAdapter:
    pytest.importorskip("torch")
    return TwoTowerAdapter(
        emb_dim=8, out_dim=8, hidden=16, epochs=2, batch_size=8, n_negatives=2
    ).fit(_dataset(), seed=0)


def test_two_tower_fit_score_schema() -> None:
    model = _model()
    user_ids = np.array([1, 2, 5])
    item_ids = np.array([10, 11, 12, 13, 14, 15])
    reco = model.score(user_ids, item_ids)

    assert set(C.Scores) <= set(reco.columns)
    assert len(reco) == len(user_ids) * len(item_ids)
    assert np.isfinite(reco[C.Score].to_numpy()).all()


def test_two_tower_unknown_ids_filtered_and_embeddings() -> None:
    model = _model()
    reco = model.score(np.array([1, 999]), np.array([10, 888]))
    assert len(reco) == 1  # only the known pair (1, 10)

    emb = model.embeddings()
    assert emb is not None
    assert emb["user"].shape == (5, 8)
    assert emb["item"].shape == (6, 8)
    assert list(emb["user_ids"]) == [1, 2, 3, 4, 5]
    assert model.get_params()["emb_dim"] == 8
