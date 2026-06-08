"""Donor model adapters. Importing submodules registers them in the ``adapters`` registry."""

from __future__ import annotations

from warmtransfer.bench.adapters.als import ALSAdapter
from warmtransfer.bench.adapters.base import ModelAdapter, adapters, register_adapter
from warmtransfer.bench.adapters.bpr import BPRAdapter
from warmtransfer.bench.adapters.catboost_adapter import CatBoostAdapter
from warmtransfer.bench.adapters.ease import EASEAdapter
from warmtransfer.bench.adapters.two_tower import TwoTowerAdapter

__all__ = [
    "ALSAdapter",
    "BPRAdapter",
    "CatBoostAdapter",
    "EASEAdapter",
    "ModelAdapter",
    "TwoTowerAdapter",
    "adapters",
    "register_adapter",
]
