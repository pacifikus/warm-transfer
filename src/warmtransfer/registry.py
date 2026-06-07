"""Generic component registry with decorator-based registration.

Used for methods (``ColdStartMethod``), donor adapters, datasets and splitters — so
that the benchmark config can reference components by string name, and adding a new
component boils down to a single decorator (the supervisor's requirement: easy
extensibility).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

from warmtransfer.exceptions import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """Registry mapping "name -> class/factory".

    Usage::

        methods: Registry[type[ColdStartMethod]] = Registry("method")

        @methods.register("grouped_most_popular")
        class GroupedMostPopular(ColdStartMethod): ...

        cls = methods.get("grouped_most_popular")
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator that registers an object under ``name``."""

        def decorator(obj: T) -> T:
            if name in self._items:
                raise RegistryError(f"{self._kind} with name {name!r} is already registered")
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> T:
        """Return the registered object by name."""
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "<empty>"
            raise RegistryError(
                f"Unknown {self._kind}: {name!r}. Registered: {known}"
            ) from None

    def names(self) -> list[str]:
        """Sorted list of registered names."""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def __len__(self) -> int:
        return len(self._items)
