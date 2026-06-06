"""Адаптеры моделей-доноров. Импорт подмодулей регистрирует их в реестре ``adapters``."""

from __future__ import annotations

from coldbench.adapters.als import ALSAdapter
from coldbench.adapters.base import ModelAdapter, adapters, register_adapter
from coldbench.adapters.bpr import BPRAdapter
from coldbench.adapters.catboost_adapter import CatBoostAdapter

__all__ = [
    "ALSAdapter",
    "BPRAdapter",
    "CatBoostAdapter",
    "ModelAdapter",
    "adapters",
    "register_adapter",
]
