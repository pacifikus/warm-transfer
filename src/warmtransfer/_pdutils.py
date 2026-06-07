"""Typed wrappers around pandas operations.

The pandas stubs produce false positives: ``df[col]`` is inferred as ``Series | DataFrame``,
and ``series.unique()``/``series.map(dict)`` as ``Unknown``. These helpers are a single
casting point at the pandas boundary, keeping call-site code clean and type-safe.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def unique_sorted(s: pd.Series | pd.DataFrame) -> np.ndarray:
    """Sorted unique values of a column as an ndarray."""
    col = cast("pd.Series", s)
    return np.sort(np.asarray(col.unique()))


def map_codes(s: pd.Series | pd.DataFrame, mapping: dict) -> np.ndarray:
    """Map column values to codes via a dictionary (ndarray)."""
    col = cast("pd.Series", s)
    return np.asarray(col.map(mapping).to_numpy())
