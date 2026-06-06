"""Загрузчик MovieLens-1M.

Контент айтема — жанры (one-hot, мульти-лейбл). Используется и для контентного сходства
cold→warm, и для popularity-в-категории (Grouped MP).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from coldbench.datasets._download import cache_dir, download, unzip
from coldbench.datasets.base import DatasetLoader, register_dataset
from coldscore.columns import Columns as C
from coldscore.types import Dataset, ItemFeatures

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML20M_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"


@register_dataset("ml-1m")
class MovieLens1M(DatasetLoader):
    """MovieLens-1M: ~6000 пользователей, ~3900 фильмов, 1M рейтингов, 18 жанров."""

    def load(self) -> Dataset:
        root = cache_dir("ml-1m")
        archive = download(ML1M_URL, root / "ml-1m.zip")
        unzip(archive, root)
        base = root / "ml-1m"

        ratings = pd.read_csv(
            base / "ratings.dat",
            sep="::",
            engine="python",
            names=[C.User, C.Item, "rating", C.Datetime],
            encoding="latin-1",
        )
        interactions = cast(
            "pd.DataFrame",
            ratings.rename(columns={"rating": C.Weight})[[C.User, C.Item, C.Weight, C.Datetime]],
        )

        movies = pd.read_csv(
            base / "movies.dat",
            sep="::",
            engine="python",
            names=[C.Item, "title", "genres"],
            encoding="latin-1",
        )
        item_features = _genres_to_features(movies)
        return Dataset(interactions=interactions, item_features=item_features, name="ml-1m")

    def describe(self) -> dict:
        return {
            "name": "ml-1m",
            "domain": "кино",
            "feedback": "explicit (рейтинги 1..5)",
            "content": "жанры (18, one-hot мульти-лейбл)",
            "url": ML1M_URL,
        }


@register_dataset("ml-20m")
class MovieLens20M(DatasetLoader):
    """MovieLens-20M: ~138k пользователей, ~27k фильмов, 20M рейтингов, 19 жанров.

    Большой кейс кино для проверки масштабируемости (тот же домен, что ML-1M).
    """

    def load(self) -> Dataset:
        root = cache_dir("ml-20m")
        archive = download(ML20M_URL, root / "ml-20m.zip")
        unzip(archive, root)
        base = root / "ml-20m"

        ratings = pd.read_csv(base / "ratings.csv")
        interactions = pd.DataFrame(
            {
                C.User: cast("pd.Series", ratings["userId"]).to_numpy(),
                C.Item: cast("pd.Series", ratings["movieId"]).to_numpy(),
                C.Weight: cast("pd.Series", ratings["rating"]).to_numpy(),
                C.Datetime: cast("pd.Series", ratings["timestamp"]).to_numpy(),
            }
        )

        movies = pd.read_csv(base / "movies.csv").rename(columns={"movieId": C.Item})
        item_features = _genres_to_features(cast("pd.DataFrame", movies))
        return Dataset(interactions=interactions, item_features=item_features, name="ml-20m")

    def describe(self) -> dict:
        return {
            "name": "ml-20m",
            "domain": "кино (большой)",
            "feedback": "explicit (рейтинги 0.5..5)",
            "content": "жанры (~19, one-hot мульти-лейбл)",
            "url": ML20M_URL,
        }


def _genres_to_features(movies: pd.DataFrame) -> ItemFeatures:
    """One-hot матрица жанров (мульти-лейбл)."""
    genre_lists = movies["genres"].str.split("|")
    all_genres = sorted({g for gl in genre_lists for g in gl})
    genre_idx = {g: j for j, g in enumerate(all_genres)}

    item_ids = movies[C.Item].to_numpy()
    matrix = np.zeros((len(movies), len(all_genres)), dtype=np.float32)
    for i, gl in enumerate(genre_lists):
        for g in gl:
            matrix[i, genre_idx[g]] = 1.0

    return ItemFeatures(item_ids=item_ids, matrix=matrix, feature_names=all_genres)
