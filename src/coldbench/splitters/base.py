"""Интерфейс сплиттера (генерация pseudo-cold) + реестр.

Сплиттер — главный носитель eval-протокола. Реализации обязаны гарантировать
инвариант отсутствия утечек (см. ``SplitResult`` и тесты).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

from coldscore.registry import Registry
from coldscore.types import Dataset, SplitResult


class Splitter(ABC):
    """Абстрактный сплиттер на warm/cold."""

    name: str = "base"

    @abstractmethod
    def split(self, dataset: Dataset, seed: int = 0) -> SplitResult:
        """Разбить датасет: выбрать pseudo-cold айтемы, убрать их из train/val."""

    def get_params(self) -> dict:
        return {}


#: Реестр сплиттеров.
splitters: Registry[type[Splitter]] = Registry("splitter")


_S = TypeVar("_S", bound=Splitter)


def register_splitter(name: str) -> Callable[[type[_S]], type[_S]]:
    def decorator(cls: type[_S]) -> type[_S]:
        cls.name = name
        splitters.register(name)(cls)
        return cls

    return decorator
