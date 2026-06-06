"""Универсальный реестр компонентов с регистрацией через декоратор.

Используется для методов (``ColdStartMethod``), адаптеров-доноров, датасетов и
сплиттеров — чтобы конфиг бенчмарка ссылался на компоненты по строковому имени, а
добавление нового компонента сводилось к одному декоратору (требование куратора —
лёгкая расширяемость).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

from coldscore.exceptions import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """Реестр «имя -> класс/фабрика».

    Использование::

        methods: Registry[type[ColdStartMethod]] = Registry("method")

        @methods.register("grouped_most_popular")
        class GroupedMostPopular(ColdStartMethod): ...

        cls = methods.get("grouped_most_popular")
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Декоратор регистрации под именем ``name``."""

        def decorator(obj: T) -> T:
            if name in self._items:
                raise RegistryError(f"{self._kind} с именем {name!r} уже зарегистрирован")
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> T:
        """Вернуть зарегистрированный объект по имени."""
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "<пусто>"
            raise RegistryError(
                f"Неизвестный {self._kind}: {name!r}. Зарегистрированы: {known}"
            ) from None

    def names(self) -> list[str]:
        """Список зарегистрированных имён (отсортированный)."""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def __len__(self) -> int:
        return len(self._items)
