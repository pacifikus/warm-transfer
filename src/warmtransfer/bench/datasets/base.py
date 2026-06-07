"""Интерфейс загрузчика датасета (DatasetLoader) + реестр.

Добавление нового датасета = один класс с ``load()`` + декоратор (требование куратора:
лёгкая расширяемость бенчмарка).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

from warmtransfer.registry import Registry
from warmtransfer.types import Dataset


class DatasetLoader(ABC):
    """Абстрактный загрузчик.

    :meth:`load` приводит сырые данные к единому формату: ``Dataset`` со взаимодействиями
    (long-format, колонки ``Columns``) и контентом айтемов (``ItemFeatures``).
    """

    name: str = "base"

    @abstractmethod
    def load(self) -> Dataset:
        """Загрузить (при необходимости скачать) и нормализовать датасет."""

    def describe(self) -> dict:
        """Краткое описание для docs/datasets.md (домен, размер, фичи)."""
        return {"name": self.name}


#: Реестр датасетов.
datasets: Registry[type[DatasetLoader]] = Registry("dataset")


_D = TypeVar("_D", bound=DatasetLoader)


def register_dataset(name: str) -> Callable[[type[_D]], type[_D]]:
    def decorator(cls: type[_D]) -> type[_D]:
        cls.name = name
        datasets.register(name)(cls)
        return cls

    return decorator
