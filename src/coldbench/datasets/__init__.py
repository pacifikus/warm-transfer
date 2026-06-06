"""Загрузчики датасетов. Импорт подмодулей регистрирует их в реестре ``datasets``."""

from __future__ import annotations

from coldbench.datasets.base import DatasetLoader, datasets, register_dataset
from coldbench.datasets.goodbooks import GoodBooks10k
from coldbench.datasets.kion import Kion, KionText
from coldbench.datasets.movielens import MovieLens1M, MovieLens20M

__all__ = [
    "DatasetLoader",
    "GoodBooks10k",
    "Kion",
    "KionText",
    "MovieLens1M",
    "MovieLens20M",
    "datasets",
    "register_dataset",
]
