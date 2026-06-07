"""Donor model adapters. Importing submodules registers them in the ``adapters`` registry."""

from __future__ import annotations

from warmtransfer.bench.adapters.als import ALSAdapter
from warmtransfer.bench.adapters.base import ModelAdapter, adapters, register_adapter
from warmtransfer.bench.adapters.bpr import BPRAdapter
from warmtransfer.bench.adapters.catboost_adapter import CatBoostAdapter

__all__ = [
    "ALSAdapter",
    "BPRAdapter",
    "CatBoostAdapter",
    "ModelAdapter",
    "adapters",
    "register_adapter",
]
