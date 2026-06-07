"""Honest pseudo-cold split by items (anti-leakage).

A subset of items is declared pseudo-cold: their interactions are COMPLETELY removed from
train (and from the val fold), remaining only in test as ground truth. The donor, Grouped MP
and neighbor search see only warm items. See ``docs/eval-protocol.md``.

Stratifying the cold sample by popularity buckets ensures that cold items span the whole
popularity range, not just the tail/head (otherwise the baselines get biased).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from warmtransfer.bench.splitters.base import Splitter, register_splitter
from warmtransfer.columns import Columns as C
from warmtransfer.schema import validate_interactions
from warmtransfer.seeding import make_rng
from warmtransfer.types import Dataset, SplitResult


def _largest_remainder(weights: list[int], target: int) -> list[int]:
    """Distribute ``target`` units across bins proportionally to ``weights``.

    Largest remainder method: floor of the proportion + handing out the remainder in
    decreasing order of fractional parts. Guarantees ``sum(result) == min(target, sum(weights))``
    and ``result[i] <= weights[i]``.
    """
    total_w = sum(weights)
    target = min(target, total_w)
    if target <= 0 or total_w == 0:
        return [0] * len(weights)

    exact = [target * w / total_w for w in weights]
    alloc = [int(e) for e in exact]
    remainder = target - sum(alloc)
    # candidates for +1 — those with spare capacity and a larger fractional part
    order = sorted(
        (i for i in range(len(weights)) if alloc[i] < weights[i]),
        key=lambda i: exact[i] - alloc[i],
        reverse=True,
    )
    for i in order[:remainder]:
        alloc[i] += 1
    return alloc


@register_splitter("pseudo_cold")
class PseudoColdSplitter(Splitter):
    """Warm / val-cold / test-cold split by items.

    :param cold_frac: fraction of items in test-cold (ground truth).
    :param val_frac: fraction of items in val-cold (hyperparameter tuning).
    :param n_pop_buckets: number of popularity buckets for stratification.
    :param min_item_interactions: minimum number of interactions for an item to become cold
        (otherwise it would have no ground truth for evaluation).
    """

    def __init__(
        self,
        cold_frac: float = 0.2,
        val_frac: float = 0.1,
        n_pop_buckets: int = 5,
        min_item_interactions: int = 1,
    ) -> None:
        if not 0 < cold_frac < 1:
            raise ValueError("cold_frac must be in (0, 1)")
        if not 0 <= val_frac < 1:
            raise ValueError("val_frac must be in [0, 1)")
        if cold_frac + val_frac >= 1:
            raise ValueError("cold_frac + val_frac must be < 1")
        self.cold_frac = cold_frac
        self.val_frac = val_frac
        self.n_pop_buckets = n_pop_buckets
        self.min_item_interactions = min_item_interactions

    def split(self, dataset: Dataset, seed: int = 0) -> SplitResult:
        inter = validate_interactions(dataset.interactions)
        rng = make_rng(seed)

        # The popularity used to stratify cold items is computed from the FULL number of
        # an item's interactions — this is deliberate and unavoidable: once an item is
        # declared cold, its interactions are removed from train, so its train-derived
        # popularity is identically zero. Stratification merely distributes cold items
        # UNIFORMLY across popularity levels (so eval is not biased toward niche/mass items)
        # and does NOT leak into method features — those see only train + static content.
        pop = cast("pd.Series", inter.groupby(C.Item).size())
        eligible = np.asarray(pop.index[pop.to_numpy() >= self.min_item_interactions])

        test_cold, val_cold = self._select_cold(eligible, pop, rng)
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

    def _select_cold(
        self, eligible: np.ndarray, pop: pd.Series, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select test-cold and val-cold items, stratified by popularity.

        The fractions are taken relative to the GLOBAL number of eligible items, then
        distributed across buckets via the largest remainder method — this prevents the
        sample from collapsing to zero under independent rounding in small strata and
        preserves the overall fraction.
        """
        pop_eligible = pop.loc[eligible]
        total = len(pop_eligible)
        if total == 0:
            return np.asarray([]), np.asarray([])

        # popularity buckets; duplicates='drop' — in case of degenerate quantiles
        try:
            buckets = pd.qcut(pop_eligible, q=self.n_pop_buckets, labels=False, duplicates="drop")
        except ValueError:
            buckets = pd.Series(0, index=pop_eligible.index)

        grouped = [grp.index.to_numpy() for _, grp in pop_eligible.groupby(buckets)]
        sizes = [len(g) for g in grouped]

        n_test_total = self._target(total, self.cold_frac)
        n_val_total = self._target(total, self.val_frac)
        # feasibility: combined total must not exceed what is available
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

    @staticmethod
    def _target(total: int, frac: float) -> int:
        """Global sample target: rounding, but ≥1 when frac>0 and the pool is non-empty."""
        if frac <= 0 or total == 0:
            return 0
        return max(1, round(total * frac))

    def get_params(self) -> dict:
        return {
            "cold_frac": self.cold_frac,
            "val_frac": self.val_frac,
            "n_pop_buckets": self.n_pop_buckets,
            "min_item_interactions": self.min_item_interactions,
        }
