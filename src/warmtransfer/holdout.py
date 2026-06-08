"""Honest pseudo-cold split by items (anti-leakage), as a pure core function.

A subset of items is declared pseudo-cold: their interactions are removed from train and
the val fold, remaining only in test as ground truth. The donor, popularity baselines and
neighbor search see only warm items. Cold items are stratified by popularity buckets so the
sample spans the whole popularity range (otherwise baselines get biased). See
``docs/eval-protocol.md``. The bench ``PseudoColdSplitter`` is a thin wrapper over this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.schema import validate_interactions
from warmtransfer.seeding import make_rng
from warmtransfer.types import Dataset, SplitResult


@dataclass
class HoldoutConfig:
    """Parameters of the pseudo-cold holdout.

    :param cold_frac: fraction of items placed in test-cold (ground truth).
    :param val_frac: fraction of items placed in val-cold (for supervised methods).
    :param n_pop_buckets: number of popularity buckets for stratification.
    :param min_item_interactions: minimum interactions for an item to be eligible as cold.
    """

    cold_frac: float = 0.2
    val_frac: float = 0.1
    n_pop_buckets: int = 5
    min_item_interactions: int = 5

    def __post_init__(self) -> None:
        if not 0 < self.cold_frac < 1:
            raise ValueError("cold_frac must be in (0, 1)")
        if not 0 <= self.val_frac < 1:
            raise ValueError("val_frac must be in [0, 1)")
        if self.cold_frac + self.val_frac >= 1:
            raise ValueError("cold_frac + val_frac must be < 1")
        if self.min_item_interactions < 0:
            raise ValueError("min_item_interactions must be >= 0")
        if self.n_pop_buckets < 1:
            raise ValueError("n_pop_buckets must be >= 1")


def _largest_remainder(weights: list[int], target: int) -> list[int]:
    """Distribute ``target`` units across bins proportionally to ``weights`` (largest remainder)."""
    total_w = sum(weights)
    target = min(target, total_w)
    if target <= 0 or total_w == 0:
        return [0] * len(weights)
    exact = [target * w / total_w for w in weights]
    alloc = [int(e) for e in exact]
    remainder = target - sum(alloc)
    order = sorted(
        (i for i in range(len(weights)) if alloc[i] < weights[i]),
        key=lambda i: exact[i] - alloc[i],
        reverse=True,
    )
    for i in order[:remainder]:
        alloc[i] += 1
    return alloc


def _target(total: int, frac: float) -> int:
    """Global sample target: rounding, but >=1 when frac>0 and the pool is non-empty."""
    if frac <= 0 or total == 0:
        return 0
    return max(1, round(total * frac))


def _select_cold(
    eligible: np.ndarray, pop: pd.Series, cfg: HoldoutConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Select test-cold and val-cold items, stratified by popularity buckets."""
    pop_eligible = pop.loc[eligible]
    total = len(pop_eligible)
    if total == 0:
        return np.asarray([]), np.asarray([])
    try:
        buckets = pd.qcut(pop_eligible, q=cfg.n_pop_buckets, labels=False, duplicates="drop")
    except ValueError:
        buckets = pd.Series(0, index=pop_eligible.index)
    grouped = [grp.index.to_numpy() for _, grp in pop_eligible.groupby(buckets)]
    sizes = [len(g) for g in grouped]

    n_test_total = _target(total, cfg.cold_frac)
    n_val_total = _target(total, cfg.val_frac)
    if n_test_total + n_val_total > total:
        n_val_total = max(0, total - n_test_total)

    test_alloc = _largest_remainder(sizes, n_test_total)
    capacity = [s - t for s, t in zip(sizes, test_alloc, strict=True)]
    val_alloc = _largest_remainder(capacity, n_val_total)

    test_cold: list = []
    val_cold: list = []
    for items, n_test, n_val in zip(grouped, test_alloc, val_alloc, strict=True):
        shuffled = items[rng.permutation(len(items))]
        test_cold.extend(shuffled[:n_test])
        val_cold.extend(shuffled[n_test : n_test + n_val])
    return np.asarray(test_cold), np.asarray(val_cold)


def pseudo_cold_split(dataset: Dataset, config: HoldoutConfig, seed: int = 0) -> SplitResult:
    """Split a dataset into warm / val-cold / test-cold by items (anti-leakage)."""
    inter = validate_interactions(dataset.interactions)
    rng = make_rng(seed)
    pop = cast("pd.Series", inter.groupby(C.Item).size())
    eligible = np.asarray(pop.index[pop.to_numpy() >= config.min_item_interactions])

    test_cold, val_cold = _select_cold(eligible, pop, config, rng)
    cold_all = list(set(test_cold.tolist()) | set(val_cold.tolist()))

    item_col = inter[C.Item]
    train = inter.loc[~item_col.isin(cold_all)].reset_index(drop=True)
    val = inter.loc[item_col.isin(val_cold.tolist())].reset_index(drop=True)
    test = inter.loc[item_col.isin(test_cold.tolist())].reset_index(drop=True)
    warm_items = np.asarray(train[C.Item].unique())

    return SplitResult(
        train=train,
        val=val,
        test=test,
        warm_items=warm_items,
        cold_items=np.asarray(sorted(test_cold)),
    )
