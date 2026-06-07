"""Общие фикстуры тестов."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from warmtransfer.columns import Columns as C
from warmtransfer.types import Dataset, ItemFeatures


@pytest.fixture
def tiny_interactions() -> pd.DataFrame:
    """Маленький набор взаимодействий (long-format)."""
    users = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4]
    items = [10, 11, 12, 10, 13, 11, 12, 14, 10, 14]
    return pd.DataFrame({C.User: users, C.Item: items, C.Weight: 1.0, C.Datetime: 0})


@pytest.fixture
def tiny_item_features() -> ItemFeatures:
    """Контент для айтемов 10..14 (5 айтемов, 3 признака)."""
    item_ids = np.array([10, 11, 12, 13, 14])
    rng = np.random.default_rng(0)
    matrix = rng.random((5, 3))
    return ItemFeatures(item_ids=item_ids, matrix=matrix, feature_names=["f0", "f1", "f2"])


@pytest.fixture
def tiny_dataset(tiny_interactions: pd.DataFrame, tiny_item_features: ItemFeatures) -> Dataset:
    return Dataset(interactions=tiny_interactions, item_features=tiny_item_features, name="tiny")
