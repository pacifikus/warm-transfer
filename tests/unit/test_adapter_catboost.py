"""CatBoostAdapter donor test: training + scoring on a small synthetic dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from warmtransfer.bench.adapters.catboost_adapter import CatBoostAdapter
from warmtransfer.columns import Columns as C
from warmtransfer.types import Dataset

pytestmark = pytest.mark.bench


def _dataset() -> Dataset:
    """5 users, 6 items."""
    users = [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5]
    items = [10, 11, 12, 10, 13, 11, 14, 12, 15, 13, 14, 15]
    inter = pd.DataFrame({C.User: users, C.Item: items, C.Weight: 1.0, C.Datetime: 0})
    return Dataset(interactions=inter, name="tiny")


def test_catboost_fit_score_schema() -> None:
    model = CatBoostAdapter(iterations=10, depth=3, neg_ratio=4).fit(_dataset(), seed=0)
    user_ids = np.array([1, 2, 5])
    item_ids = np.array([10, 11, 12, 13, 14, 15])
    reco = model.score(user_ids, item_ids)

    # Output schema.
    assert set(C.Scores) <= set(reco.columns)
    # Cross product of known pairs.
    assert len(reco) == len(user_ids) * len(item_ids)
    # Score range [0, 1].
    scores = reco[C.Score].to_numpy()
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_catboost_unknown_ids_filtered() -> None:
    model = CatBoostAdapter(iterations=10, depth=3, neg_ratio=4).fit(_dataset(), seed=0)
    reco = model.score(np.array([1, 999]), np.array([10, 888]))
    # Only the known pair (1, 10) remains.
    assert len(reco) == 1
    assert model.embeddings() is None
    assert model.get_params() == {"iterations": 10, "depth": 3, "neg_ratio": 4}
