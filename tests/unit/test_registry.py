"""Тесты реестра компонентов."""

from __future__ import annotations

import pytest

from coldscore.exceptions import RegistryError
from coldscore.registry import Registry


def test_register_and_get() -> None:
    reg: Registry[type] = Registry("thing")

    @reg.register("foo")
    class Foo: ...

    assert reg.get("foo") is Foo
    assert "foo" in reg
    assert reg.names() == ["foo"]
    assert len(reg) == 1


def test_duplicate_name_raises() -> None:
    reg: Registry[type] = Registry("thing")

    @reg.register("foo")
    class Foo: ...

    with pytest.raises(RegistryError, match="уже зарегистрирован"):

        @reg.register("foo")
        class Bar: ...


def test_unknown_name_raises() -> None:
    reg: Registry[type] = Registry("thing")
    with pytest.raises(RegistryError, match="Неизвестный thing"):
        reg.get("missing")
