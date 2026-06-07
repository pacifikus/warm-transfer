"""Загрузчик GoodBooks-10k.

Контент айтема (книги): one-hot топ-200 авторов (первый автор) + one-hot бакета
десятилетия публикации. Используется для контентного сходства cold→warm.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from warmtransfer.bench.datasets._download import cache_dir, download
from warmtransfer.bench.datasets.base import DatasetLoader, register_dataset
from warmtransfer.columns import Columns as C
from warmtransfer.types import Dataset, ItemFeatures

RATINGS_URL = (
    "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv"
)
BOOKS_URL = (
    "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"
)

#: Сколько самых частых авторов оставляем в one-hot.
TOP_AUTHORS = 200


@register_dataset("goodbooks")
class GoodBooks10k(DatasetLoader):
    """GoodBooks-10k: ~53k пользователей, 10k книг, ~6M рейтингов (1..5)."""

    def load(self) -> Dataset:
        root = cache_dir("goodbooks")
        ratings_path = download(RATINGS_URL, root / "ratings.csv")
        books_path = download(BOOKS_URL, root / "books.csv")

        ratings = pd.read_csv(ratings_path)
        renamed = ratings.rename(columns={"book_id": C.Item, "rating": C.Weight})
        interactions = cast("pd.DataFrame", renamed[[C.User, C.Item, C.Weight]]).copy()
        interactions[C.Datetime] = 0

        books = pd.read_csv(books_path)
        item_features = _books_to_features(books)
        return Dataset(
            interactions=interactions,
            item_features=item_features,
            name="goodbooks",
        )

    def describe(self) -> dict:
        return {
            "name": "goodbooks",
            "domain": "книги",
            "feedback": "explicit (рейтинги 1..5)",
            "size": "~53k пользователей, 10k книг, ~6M рейтингов",
            "content": (
                f"one-hot топ-{TOP_AUTHORS} авторов (первый автор) + "
                "one-hot бакета десятилетия публикации"
            ),
            "ratings_url": RATINGS_URL,
            "books_url": BOOKS_URL,
        }


def _first_author(authors: object) -> str:
    """Первый автор (до запятой). Пропуски → пустая строка."""
    if not isinstance(authors, str):
        return ""
    return authors.split(",")[0].strip()


def _decade_bucket(year: object) -> str:
    """Бакет десятилетия по году публикации. Пропуски → отдельная категория."""
    if year is None or (isinstance(year, float) and np.isnan(year)):
        return "decade_unknown"
    try:
        y = int(float(cast("float", year)))
    except (ValueError, TypeError):
        return "decade_unknown"
    return f"decade_{y // 10 * 10}"


def _books_to_features(books: pd.DataFrame) -> ItemFeatures:
    """One-hot топ-N авторов (первый автор) + one-hot бакета десятилетия.

    Объединяет обе группы признаков в одну float-матрицу. ``feature_names``
    содержит сначала имена авторов (с префиксом ``author=``), затем десятилетия.
    """
    item_ids = np.asarray(cast("pd.Series", books["book_id"]).to_numpy())
    n = len(books)

    authors_col = cast("pd.Series", books["authors"])
    first_authors = [_first_author(a) for a in authors_col.tolist()]

    # топ-N авторов по частоте (исключаем пустые)
    counts = pd.Series([a for a in first_authors if a]).value_counts()
    ranked = cast("list[str]", counts.index.tolist())
    top_authors: list[str] = [str(a) for a in ranked[:TOP_AUTHORS]]
    author_idx = {a: j for j, a in enumerate(top_authors)}

    year_col = cast("pd.Series", books["original_publication_year"])
    decades = [_decade_bucket(y) for y in year_col.tolist()]
    all_decades = sorted(set(decades))
    decade_idx = {d: j for j, d in enumerate(all_decades)}

    n_authors = len(top_authors)
    n_decades = len(all_decades)
    matrix = np.zeros((n, n_authors + n_decades), dtype=np.float32)

    for i in range(n):
        a = first_authors[i]
        if a in author_idx:
            matrix[i, author_idx[a]] = 1.0
        matrix[i, n_authors + decade_idx[decades[i]]] = 1.0

    feature_names = [f"author={a}" for a in top_authors] + list(all_decades)
    return ItemFeatures(
        item_ids=item_ids,
        matrix=matrix,
        feature_names=feature_names,
    )
