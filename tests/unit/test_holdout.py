"""Tests for the core pure-function pseudo-cold split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.holdout import HoldoutConfig, pseudo_cold_split
from warmtransfer.types import Dataset


def _toy_dataset(n_users: int = 30, n_items: int = 20, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    rows = []
    for it in range(n_items):
        # popularity grows with item id so qcut buckets are non-degenerate
        for u in rng.choice(n_users, size=2 + it % 5, replace=False):
            rows.append((int(u), it))
    df = pd.DataFrame(rows, columns=[C.User, C.Item])
    return Dataset(interactions=df, name="toy")


def test_split_no_leak() -> None:
    ds = _toy_dataset()
    split = pseudo_cold_split(ds, HoldoutConfig(cold_frac=0.2, val_frac=0.1), seed=42)
    assert set(split.cold_items) & set(split.train[C.Item].unique()) == set()
    assert set(split.cold_items) & set(split.val[C.Item].unique()) == set()
    assert set(split.test[C.Item].unique()) == set(split.cold_items)


def test_split_deterministic() -> None:
    ds = _toy_dataset()
    a = pseudo_cold_split(ds, HoldoutConfig(), seed=7)
    b = pseudo_cold_split(ds, HoldoutConfig(), seed=7)
    assert sorted(a.cold_items) == sorted(b.cold_items)


def test_holdout_config_defaults() -> None:
    cfg = HoldoutConfig()
    assert cfg.cold_frac == 0.2 and cfg.val_frac == 0.1
    assert cfg.n_pop_buckets == 5 and cfg.min_item_interactions == 5
