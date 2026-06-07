"""Тест донора BPRAdapter: обучение + скоринг на маленьком synthetic-датасете."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from warmtransfer.bench.adapters.bpr import BPRAdapter
from warmtransfer.columns import Columns as C
from warmtransfer.types import Dataset

pytestmark = pytest.mark.bench


def _dataset() -> Dataset:
    """5 пользователей, 6 айтемов."""
    users = [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5]
    items = [10, 11, 12, 10, 13, 11, 14, 12, 15, 13, 14, 15]
    inter = pd.DataFrame({C.User: users, C.Item: items, C.Weight: 1.0, C.Datetime: 0})
    return Dataset(interactions=inter, name="tiny")


def test_bpr_fit_score_schema() -> None:
    model = BPRAdapter(factors=8, iterations=10).fit(_dataset(), seed=0)
    user_ids = np.array([1, 2, 5])
    item_ids = np.array([10, 11, 12, 13, 14, 15])
    reco = model.score(user_ids, item_ids)

    assert set(C.Scores) <= set(reco.columns)
    assert len(reco) == len(user_ids) * len(item_ids)
    assert np.isfinite(reco[C.Score].to_numpy()).all()


def test_bpr_unknown_ids_filtered_and_embeddings() -> None:
    model = BPRAdapter(factors=8, iterations=10).fit(_dataset(), seed=0)
    reco = model.score(np.array([1, 999]), np.array([10, 888]))
    assert len(reco) == 1  # только известная пара (1, 10)

    emb = model.embeddings()
    assert emb is not None
    assert emb["user"].shape[0] == 5
    assert emb["item"].shape[0] == 6
    assert model.get_params()["factors"] == 8
