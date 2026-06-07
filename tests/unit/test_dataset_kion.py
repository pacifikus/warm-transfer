"""Тест построения контент-фич KION из синтетического items-фрейма (без загрузки)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.bench.datasets.kion import _items_to_features, _split_list


def test_split_list_normalizes() -> None:
    assert _split_list("Drama, Foreign, Detective") == ["drama", "foreign", "detective"]
    assert _split_list(None) == []
    assert _split_list(float("nan")) == []


def test_items_to_features_multihot() -> None:
    items = pd.DataFrame(
        {
            "item_id": [101, 102, 103],
            "content_type": ["film", "series", "film"],
            "release_year": [2002.0, 2014.0, np.nan],
            "genres": ["drama, foreign", "comedy", "drama, comedy"],
            "countries": ["Spain", "USA", "USA"],
        }
    )
    feats = _items_to_features(items)
    mat = np.asarray(feats.matrix)

    assert list(feats.item_ids) == [101, 102, 103]
    assert mat.shape[0] == 3
    names = feats.feature_names

    # жанры — multi-hot: item 103 (drama, comedy) активирует оба жанра
    drama = names.index("genre=drama")
    comedy = names.index("genre=comedy")
    assert mat[2, drama] == 1.0
    assert mat[2, comedy] == 1.0
    assert mat[1, drama] == 0.0

    # тип контента — one-hot
    assert mat[0, names.index("type=film")] == 1.0
    assert mat[1, names.index("type=series")] == 1.0

    # десятилетие: пропущенный год → decade_unknown
    assert mat[2, names.index("decade_unknown")] == 1.0
    assert mat[0, names.index("decade_2000")] == 1.0
