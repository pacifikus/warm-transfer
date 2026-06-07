"""Dataset loader interface (DatasetLoader) + registry.

Adding a new dataset = one class with ``load()`` + a decorator (curator requirement:
easy extensibility of the benchmark).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

from warmtransfer.registry import Registry
from warmtransfer.types import Dataset


class DatasetLoader(ABC):
    """Abstract loader.

    :meth:`load` converts raw data into a unified format: a ``Dataset`` with interactions
    (long-format, ``Columns`` columns) and item content (``ItemFeatures``).
    """

    name: str = "base"

    @abstractmethod
    def load(self) -> Dataset:
        """Load (downloading if necessary) and normalize the dataset."""

    def describe(self) -> dict:
        """Brief description for docs/datasets.md (domain, size, features)."""
        return {"name": self.name}


#: Dataset registry.
datasets: Registry[type[DatasetLoader]] = Registry("dataset")


_D = TypeVar("_D", bound=DatasetLoader)


def register_dataset(name: str) -> Callable[[type[_D]], type[_D]]:
    def decorator(cls: type[_D]) -> type[_D]:
        cls.name = name
        datasets.register(name)(cls)
        return cls

    return decorator
