"""Честный pseudo-cold сплит по айтемам (анти-утечка).

Подмножество айтемов объявляется псевдо-cold: их взаимодействия ПОЛНОСТЬЮ убираются из
train (и из val-фолда), оставаясь только в test как ground truth. Донор, Grouped MP и
поиск соседей видят только warm-айтемы. См. ``docs/eval-protocol.md``.

Стратификация cold-выборки по бакетам популярности — чтобы cold-айтемы покрывали весь
диапазон популярности, а не только хвост/голову (иначе бейзлайны смещаются).
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
    """Распределить ``target`` единиц по корзинам пропорционально ``weights``.

    Метод наибольшего остатка: floor от пропорции + раздача остатка по убыванию
    дробных частей. Гарантирует ``sum(result) == min(target, sum(weights))`` и
    ``result[i] <= weights[i]``.
    """
    total_w = sum(weights)
    target = min(target, total_w)
    if target <= 0 or total_w == 0:
        return [0] * len(weights)

    exact = [target * w / total_w for w in weights]
    alloc = [int(e) for e in exact]
    remainder = target - sum(alloc)
    # кандидаты на +1 — у кого есть запас и больше дробная часть
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
    """Сплит warm / val-cold / test-cold по айтемам.

    :param cold_frac: доля айтемов в test-cold (ground truth).
    :param val_frac: доля айтемов в val-cold (тюнинг гиперпараметров).
    :param n_pop_buckets: число бакетов популярности для стратификации.
    :param min_item_interactions: минимум взаимодействий, чтобы айтем мог стать cold
        (иначе у него не будет ground truth для оценки).
    """

    def __init__(
        self,
        cold_frac: float = 0.2,
        val_frac: float = 0.1,
        n_pop_buckets: int = 5,
        min_item_interactions: int = 1,
    ) -> None:
        if not 0 < cold_frac < 1:
            raise ValueError("cold_frac должен быть в (0, 1)")
        if not 0 <= val_frac < 1:
            raise ValueError("val_frac должен быть в [0, 1)")
        if cold_frac + val_frac >= 1:
            raise ValueError("cold_frac + val_frac должны быть < 1")
        self.cold_frac = cold_frac
        self.val_frac = val_frac
        self.n_pop_buckets = n_pop_buckets
        self.min_item_interactions = min_item_interactions

    def split(self, dataset: Dataset, seed: int = 0) -> SplitResult:
        inter = validate_interactions(dataset.interactions)
        rng = make_rng(seed)

        # Популярность для стратификации cold-айтемов считается по ПОЛНОМУ числу
        # взаимодействий айтема — это намеренно и неизбежно: после объявления айтема
        # cold его взаимодействия удаляются из train, поэтому популярность из train
        # для него тождественно нулевая. Стратификация лишь РАВНОМЕРНО распределяет
        # cold-айтемы по уровням популярности (чтобы eval не смещался к нишевым/массовым)
        # и НЕ протекает в признаки методов — те видят только train + статичный контент.
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
        """Стратифицированно по популярности выбрать test-cold и val-cold айтемы.

        Доли считаются от ГЛОБАЛЬНОГО числа eligible-айтемов, затем распределяются по
        бакетам методом наибольшего остатка (largest remainder) — это исключает обнуление
        выборки при независимом округлении в малых стратах и сохраняет общую долю.
        """
        pop_eligible = pop.loc[eligible]
        total = len(pop_eligible)
        if total == 0:
            return np.asarray([]), np.asarray([])

        # бакеты популярности; duplicates='drop' — на случай вырожденных квантилей
        try:
            buckets = pd.qcut(pop_eligible, q=self.n_pop_buckets, labels=False, duplicates="drop")
        except ValueError:
            buckets = pd.Series(0, index=pop_eligible.index)

        grouped = [grp.index.to_numpy() for _, grp in pop_eligible.groupby(buckets)]
        sizes = [len(g) for g in grouped]

        n_test_total = self._target(total, self.cold_frac)
        n_val_total = self._target(total, self.val_frac)
        # феасибилити: суммарно не больше доступного
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
        """Глобальный target выборки: округление, но ≥1 при frac>0 и непустом пуле."""
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
