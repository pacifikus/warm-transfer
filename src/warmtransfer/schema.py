"""Validation and normalization of input DataFrames.

DataFrames map poorly onto pydantic models, so here we use explicit validator functions
that raise :class:`SchemaError`. Pydantic is used for configs
(see ``warmtransfer.bench.config``).
"""

from __future__ import annotations

import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.exceptions import SchemaError


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], where: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{where}: missing columns {missing}. Present: {list(df.columns)}")


def validate_interactions(df: pd.DataFrame, *, require_weight: bool = False) -> pd.DataFrame:
    """Validate and normalize an interactions DataFrame.

    Requires ``user_id``, ``item_id``. Adds ``weight=1.0`` if it is missing and
    ``require_weight=False``. Raises :class:`SchemaError` on duplicate pairs or NaN
    in the key columns.
    """
    _require_columns(df, C.Interactions, "interactions")
    out = df.copy()

    if bool(out[[C.User, C.Item]].isna().to_numpy().any()):
        raise SchemaError("interactions: NaN in user_id/item_id")

    if C.Weight not in out.columns:
        if require_weight:
            raise SchemaError("interactions: weight column is required")
        out[C.Weight] = 1.0

    dup = out.duplicated(subset=[C.User, C.Item]).sum()
    if dup:
        raise SchemaError(f"interactions: {dup} duplicate (user_id, item_id) pairs")

    return out


def validate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a donor scores DataFrame: ``user_id``, ``item_id``, ``score``."""
    _require_columns(df, C.Scores, "scores")
    out = df.copy()
    if bool(out[C.Score].isna().any()):
        raise SchemaError("scores: NaN in score column")
    dup = out.duplicated(subset=[C.User, C.Item]).sum()
    if dup:
        raise SchemaError(f"scores: {dup} duplicate (user_id, item_id) pairs")
    return out


def validate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a method output: ``user_id``, ``item_id``, ``score`` (+ optional ``rank``)."""
    _require_columns(df, C.Scores, "recommendations")
    if bool(df[C.Score].isna().any()):
        raise SchemaError("recommendations: NaN in score column")
    return df
