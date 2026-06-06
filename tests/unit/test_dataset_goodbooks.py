"""Юнит-тест для GoodBooks-10k: проверяем построение фич без скачивания."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from coldbench.datasets.goodbooks import _books_to_features
from coldscore.types import ItemFeatures


def _synthetic_books() -> pd.DataFrame:
    """Маленький synthetic books DataFrame."""
    return pd.DataFrame(
        {
            "book_id": [1, 2, 3, 4],
            "authors": [
                "Jane Austen, Editor X",  # первый автор — Jane Austen
                "Jane Austen",  # тот же первый автор
                "Leo Tolstoy",
                None,  # пропуск автора
            ],
            "original_publication_year": [1813.0, 1815.0, 1869.0, np.nan],
        }
    )


def test_books_to_features_type_and_shape() -> None:
    feats = _books_to_features(_synthetic_books())
    mat = cast("np.ndarray", feats.matrix)

    assert isinstance(feats, ItemFeatures)
    assert feats.n_items == 4
    assert mat.shape[0] == 4
    # число колонок = число авторов + число десятилетий
    assert mat.shape[1] == len(feats.feature_names)
    assert mat.dtype == np.float32
    np.testing.assert_array_equal(feats.item_ids, np.array([1, 2, 3, 4]))


def test_books_to_features_author_one_hot() -> None:
    feats = _books_to_features(_synthetic_books())
    mat = cast("np.ndarray", feats.matrix)

    cols = feats.feature_names
    austen_col = cols.index("author=Jane Austen")
    # книги 0 и 1 имеют первого автора Jane Austen
    assert mat[0, austen_col] == 1.0
    assert mat[1, austen_col] == 1.0
    # книга 2 (Tolstoy) — нет
    assert mat[2, austen_col] == 0.0


def test_books_to_features_decade_one_hot() -> None:
    feats = _books_to_features(_synthetic_books())
    mat = cast("np.ndarray", feats.matrix)

    cols = feats.feature_names
    # 1813 -> decade_1810, 1869 -> decade_1860, NaN -> decade_unknown
    d1810 = cols.index("decade_1810")
    d1860 = cols.index("decade_1860")
    d_unknown = cols.index("decade_unknown")

    assert mat[0, d1810] == 1.0
    assert mat[2, d1860] == 1.0
    assert mat[3, d_unknown] == 1.0
    # у книги без года остальные десятилетия выставлены в 0
    assert mat[3, d1810] == 0.0
