"""Исключения библиотеки."""

from __future__ import annotations


class ColdScoreError(Exception):
    """Базовое исключение библиотеки."""


class SchemaError(ColdScoreError):
    """Нарушение схемы входного DataFrame (нет колонки, неверный dtype, дубликаты, NaN)."""


class MissingInputError(ColdScoreError):
    """Методу не передан обязательный вход (см. ``ColdStartMethod.requires``)."""


class NotFittedError(ColdScoreError):
    """Вызов ``predict`` до ``fit``."""


class RegistryError(ColdScoreError):
    """Ошибка реестра компонентов (дубликат имени, неизвестное имя)."""


class LeakageError(ColdScoreError):
    """Обнаружена утечка cold-айтемов в обучающие данные (нарушение eval-протокола)."""
