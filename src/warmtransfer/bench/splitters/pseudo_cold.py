"""Honest pseudo-cold split by items — bench wrapper over ``warmtransfer.holdout``.

The actual logic lives in core (``warmtransfer.holdout.pseudo_cold_split``) so the
high-level ``recommend()`` can reuse it without the bench extra. This class preserves the
``Splitter``/registry contract used by the benchmark runner and YAML configs.
"""

from __future__ import annotations

from warmtransfer.bench.splitters.base import Splitter, register_splitter
from warmtransfer.holdout import HoldoutConfig, pseudo_cold_split
from warmtransfer.types import Dataset, SplitResult


@register_splitter("pseudo_cold")
class PseudoColdSplitter(Splitter):
    """Warm / val-cold / test-cold split by items (logic in ``warmtransfer.holdout``).

    :param cold_frac: fraction of items in test-cold (ground truth).
    :param val_frac: fraction of items in val-cold (hyperparameter tuning).
    :param n_pop_buckets: number of popularity buckets for stratification.
    :param min_item_interactions: min interactions for an item to be eligible as cold.
    """

    def __init__(
        self,
        cold_frac: float = 0.2,
        val_frac: float = 0.1,
        n_pop_buckets: int = 5,
        # дефолт 1 намеренно отличается от HoldoutConfig (5) ради обратной совместимости bench
        min_item_interactions: int = 1,
    ) -> None:
        self._cfg = HoldoutConfig(
            cold_frac=cold_frac,
            val_frac=val_frac,
            n_pop_buckets=n_pop_buckets,
            min_item_interactions=min_item_interactions,
        )

    def split(self, dataset: Dataset, seed: int = 0) -> SplitResult:
        return pseudo_cold_split(dataset, self._cfg, seed)

    def get_params(self) -> dict:
        return {
            "cold_frac": self._cfg.cold_frac,
            "val_frac": self._cfg.val_frac,
            "n_pop_buckets": self._cfg.n_pop_buckets,
            "min_item_interactions": self._cfg.min_item_interactions,
        }
