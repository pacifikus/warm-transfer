"""Типизированные обёртки над pandas-операциями.

Стабы pandas дают ложные срабатывания: ``df[col]`` выводится как ``Series | DataFrame``,
а ``series.unique()``/``series.map(dict)`` — как ``Unknown``. Эти хелперы — единая точка
каста на границе с pandas, чтобы код вызова оставался чистым и типобезопасным.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def unique_sorted(s: pd.Series | pd.DataFrame) -> np.ndarray:
    """Отсортированные уникальные значения колонки как ndarray."""
    col = cast("pd.Series", s)
    return np.sort(np.asarray(col.unique()))


def map_codes(s: pd.Series | pd.DataFrame, mapping: dict) -> np.ndarray:
    """Отобразить значения колонки в коды по словарю (ndarray)."""
    col = cast("pd.Series", s)
    return np.asarray(col.map(mapping).to_numpy())
