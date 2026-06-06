"""Валидация и нормализация входных DataFrame.

DataFrame'ы плохо ложатся на pydantic-модели, поэтому здесь — явные функции-валидаторы,
бросающие :class:`SchemaError`. Pydantic используется для конфигов (см. ``coldbench.config``).
"""

from __future__ import annotations

import pandas as pd

from coldscore.columns import Columns as C
from coldscore.exceptions import SchemaError


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], where: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{where}: отсутствуют колонки {missing}. Есть: {list(df.columns)}")


def validate_interactions(df: pd.DataFrame, *, require_weight: bool = False) -> pd.DataFrame:
    """Проверить и нормализовать DataFrame взаимодействий.

    Требует ``user_id``, ``item_id``. Добавляет ``weight=1.0``, если её нет и
    ``require_weight=False``. Бросает :class:`SchemaError` при дубликатах пар или NaN
    в ключевых колонках.
    """
    _require_columns(df, C.Interactions, "interactions")
    out = df.copy()

    if bool(out[[C.User, C.Item]].isna().to_numpy().any()):
        raise SchemaError("interactions: NaN в user_id/item_id")

    if C.Weight not in out.columns:
        if require_weight:
            raise SchemaError("interactions: требуется колонка weight")
        out[C.Weight] = 1.0

    dup = out.duplicated(subset=[C.User, C.Item]).sum()
    if dup:
        raise SchemaError(f"interactions: {dup} дублирующихся пар (user_id, item_id)")

    return out


def validate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Проверить DataFrame скоров донора: ``user_id``, ``item_id``, ``score``."""
    _require_columns(df, C.Scores, "scores")
    out = df.copy()
    if bool(out[C.Score].isna().any()):
        raise SchemaError("scores: NaN в колонке score")
    dup = out.duplicated(subset=[C.User, C.Item]).sum()
    if dup:
        raise SchemaError(f"scores: {dup} дублирующихся пар (user_id, item_id)")
    return out


def validate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Проверить выход метода: ``user_id``, ``item_id``, ``score`` (+ опц. ``rank``)."""
    _require_columns(df, C.Scores, "recommendations")
    if bool(df[C.Score].isna().any()):
        raise SchemaError("recommendations: NaN в колонке score")
    return df
