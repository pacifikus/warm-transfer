"""Donor model interface (ModelAdapter) + registry.

The adapter wraps a third-party model (implicit/LightFM/CatBoost/RecTools) into a
unified interface: train on warm interactions and emit scores for warm items. These
scores are the input for a cold-start method. The transfer itself is done by
``warmtransfer``, not by the adapter.
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
    """Abstract donor.

    Contract:
      * :meth:`fit` trains ONLY on warm interactions (no cold items there);
      * :meth:`score` returns ``[user_id, item_id, score]`` for the requested pairs;
      * :meth:`embeddings` — optional user/item latent factors (for [EMB] methods).
    """

    name: str = "base"

    @abstractmethod
    def fit(self, dataset: Dataset, seed: int = 0) -> ModelAdapter:
        """Train the donor on ``dataset.interactions`` (warm)."""

    @abstractmethod
    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> pd.DataFrame:
        """Scores for pairs (user_ids × item_ids), long-format ``[user_id, item_id, score]``."""

    def embeddings(self) -> dict[str, np.ndarray] | None:
        """Latent factors, if the model has them: ``{"user": ..., "item": ...}``."""
        return None

    def get_params(self) -> dict:
        return {}


#: Donor registry.
adapters: Registry[type[ModelAdapter]] = Registry("adapter")


_A = TypeVar("_A", bound=ModelAdapter)


def register_adapter(name: str) -> Callable[[type[_A]], type[_A]]:
    def decorator(cls: type[_A]) -> type[_A]:
        cls.name = name
        adapters.register(name)(cls)
        return cls

    return decorator
