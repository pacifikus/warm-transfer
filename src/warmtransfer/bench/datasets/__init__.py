"""Загрузчики датасетов. Импорт подмодулей регистрирует их в реестре ``datasets``."""

from __future__ import annotations

from warmtransfer.bench.datasets.base import DatasetLoader, datasets, register_dataset
from warmtransfer.bench.datasets.goodbooks import GoodBooks10k
from warmtransfer.bench.datasets.kion import Kion, KionText
from warmtransfer.bench.datasets.movielens import MovieLens1M, MovieLens20M

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
