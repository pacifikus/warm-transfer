"""Сплиттеры. Импорт подмодулей регистрирует их в реестре ``splitters``."""

from __future__ import annotations

from warmtransfer.bench.splitters.base import Splitter, register_splitter, splitters
from warmtransfer.bench.splitters.pseudo_cold import PseudoColdSplitter

__all__ = ["PseudoColdSplitter", "Splitter", "register_splitter", "splitters"]
