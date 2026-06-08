"""Dataset loaders. Importing the submodules registers them in the ``datasets`` registry."""

from __future__ import annotations

from warmtransfer.bench.datasets.amazon import AmazonToys, AmazonToysText
from warmtransfer.bench.datasets.base import DatasetLoader, datasets, register_dataset
from warmtransfer.bench.datasets.goodbooks import GoodBooks10k
from warmtransfer.bench.datasets.kion import Kion, KionText
from warmtransfer.bench.datasets.mind import Mind, MindText
from warmtransfer.bench.datasets.movielens import MovieLens1M, MovieLens20M

__all__ = [
    "AmazonToys",
    "AmazonToysText",
    "DatasetLoader",
    "GoodBooks10k",
    "Kion",
    "KionText",
    "Mind",
    "MindText",
    "MovieLens1M",
    "MovieLens20M",
    "datasets",
    "register_dataset",
]
