"""Splitter interface (pseudo-cold generation) + registry.

The splitter is the main carrier of the eval protocol. Implementations must
guarantee the no-leakage invariant (see ``SplitResult`` and the tests).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

from warmtransfer.registry import Registry
from warmtransfer.types import Dataset, SplitResult


class Splitter(ABC):
    """Abstract warm/cold splitter."""

    name: str = "base"

    @abstractmethod
    def split(self, dataset: Dataset, seed: int = 0) -> SplitResult:
        """Split the dataset: pick pseudo-cold items and remove them from train/val."""

    def get_params(self) -> dict:
        return {}


#: Splitter registry.
splitters: Registry[type[Splitter]] = Registry("splitter")


_S = TypeVar("_S", bound=Splitter)


def register_splitter(name: str) -> Callable[[type[_S]], type[_S]]:
    def decorator(cls: type[_S]) -> type[_S]:
        cls.name = name
        splitters.register(name)(cls)
        return cls

    return decorator
