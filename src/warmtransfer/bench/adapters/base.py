"""Интерфейс модели-донора (ModelAdapter) + реестр.

Адаптер оборачивает стороннюю модель (implicit/LightFM/CatBoost/RecTools) в единый
интерфейс: обучить на warm-взаимодействиях и выдать скоры по warm-айтемам. Эти скоры
— вход для cold-start метода. Сам трансфер делает ``warmtransfer``, не адаптер.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

import numpy as np
import pandas as pd

from warmtransfer.registry import Registry
from warmtransfer.types import Dataset


class ModelAdapter(ABC):
    """Абстрактный донор.

    Контракт:
      * :meth:`fit` обучается ТОЛЬКО на warm-взаимодействиях (cold-айтемов там нет);
      * :meth:`score` возвращает ``[user_id, item_id, score]`` по запрошенным парам;
      * :meth:`embeddings` — опционально user/item латентные факторы (для [EMB]-методов).
    """

    name: str = "base"

    @abstractmethod
    def fit(self, dataset: Dataset, seed: int = 0) -> ModelAdapter:
        """Обучить донора на ``dataset.interactions`` (warm)."""

    @abstractmethod
    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> pd.DataFrame:
        """Скоры по парам (user_ids × item_ids), long-format ``[user_id, item_id, score]``."""

    def embeddings(self) -> dict[str, np.ndarray] | None:
        """Латентные факторы, если модель их имеет: ``{"user": ..., "item": ...}``."""
        return None

    def get_params(self) -> dict:
        return {}


#: Реестр доноров.
adapters: Registry[type[ModelAdapter]] = Registry("adapter")


_A = TypeVar("_A", bound=ModelAdapter)


def register_adapter(name: str) -> Callable[[type[_A]], type[_A]]:
    def decorator(cls: type[_A]) -> type[_A]:
        cls.name = name
        adapters.register(name)(cls)
        return cls

    return decorator
