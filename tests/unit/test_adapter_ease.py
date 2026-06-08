"""Test for the EASEAdapter donor: training + scoring on a small synthetic dataset.

EASE is a deterministic closed-form linear item-item model and has NO latent space, so
``embeddings()`` must return ``None`` (it is not an [EMB]-capable donor).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from warmtransfer.bench.adapters.ease import EASEAdapter
from warmtransfer.columns import Columns as C
from warmtransfer.types import Dataset

pytestmark = pytest.mark.bench


def _dataset() -> Dataset:
    """5 users, 6 items."""
    users = [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5]
    items = [10, 11, 12, 10, 13, 11, 14, 12, 15, 13, 14, 15]
    inter = pd.DataFrame({C.User: users, C.Item: items, C.Weight: 1.0, C.Datetime: 0})
    return Dataset(interactions=inter, name="tiny")


def test_ease_fit_score_schema() -> None:
    model = EASEAdapter(l2=10.0).fit(_dataset(), seed=0)
    user_ids = np.array([1, 2, 5])
    item_ids = np.array([10, 11, 12, 13, 14, 15])
    reco = model.score(user_ids, item_ids)

    assert set(C.Scores) <= set(reco.columns)
    assert len(reco) == len(user_ids) * len(item_ids)
    assert np.isfinite(reco[C.Score].to_numpy()).all()


def test_ease_unknown_ids_filtered_no_embeddings() -> None:
    model = EASEAdapter(l2=10.0).fit(_dataset(), seed=0)
    reco = model.score(np.array([1, 999]), np.array([10, 888]))
    assert len(reco) == 1  # only the known pair (1, 10)

    # EASE has no latent space -> no embeddings.
    assert model.embeddings() is None
    assert model.get_params()["l2"] == 10.0


def test_ease_deterministic() -> None:
    """Closed-form solution -> identical scores across seeds."""
    a = EASEAdapter(l2=10.0).fit(_dataset(), seed=0)
    b = EASEAdapter(l2=10.0).fit(_dataset(), seed=123)
    users, items = np.array([1, 2, 3]), np.array([10, 11, 12])
    sa = a.score(users, items)[C.Score].to_numpy()
    sb = b.score(users, items)[C.Score].to_numpy()
    assert np.allclose(sa, sb)
