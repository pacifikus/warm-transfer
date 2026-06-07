"""Тесты валидации/нормализации входных DataFrame."""

from __future__ import annotations

import pandas as pd
import pytest

from warmtransfer.columns import Columns as C
from warmtransfer.exceptions import SchemaError
from warmtransfer.schema import validate_interactions, validate_scores


def test_validate_interactions_adds_weight() -> None:
    df = pd.DataFrame({C.User: [1, 2], C.Item: [10, 11]})
    out = validate_interactions(df)
    assert C.Weight in out.columns
    assert (out[C.Weight] == 1.0).all()


def test_validate_interactions_missing_column() -> None:
    df = pd.DataFrame({C.User: [1]})
    with pytest.raises(SchemaError, match="отсутствуют колонки"):
        validate_interactions(df)


def test_validate_interactions_duplicates() -> None:
    df = pd.DataFrame({C.User: [1, 1], C.Item: [10, 10]})
    with pytest.raises(SchemaError, match="дублирующихся пар"):
        validate_interactions(df)


def test_validate_scores_nan() -> None:
    df = pd.DataFrame({C.User: [1], C.Item: [10], C.Score: [float("nan")]})
    with pytest.raises(SchemaError, match="NaN"):
        validate_scores(df)
