"""Тесты честного pseudo-cold сплита — главный анти-утечка контракт.

Если эти тесты падают — eval-протокол скомпрометирован, всем цифрам нельзя верить.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from warmtransfer.bench.splitters.pseudo_cold import PseudoColdSplitter
from warmtransfer.columns import Columns as C
from warmtransfer.types import Dataset


@pytest.fixture
def medium_dataset() -> Dataset:
    """200 пользователей × 80 айтемов, популярность распределена неравномерно."""
    rng = np.random.default_rng(0)
    users, items = [], []
    for item in range(80):
        # популярность айтема убывает с индексом
        n = max(2, int(60 * np.exp(-item / 20)))
        chosen = rng.choice(200, size=min(n, 200), replace=False)
        users.extend(chosen.tolist())
        items.extend([item] * len(chosen))
    inter = pd.DataFrame({C.User: users, C.Item: items, C.Weight: 1.0, C.Datetime: 0})
    return Dataset(interactions=inter, name="medium")


def test_no_cold_item_in_train(medium_dataset: Dataset) -> None:
    res = PseudoColdSplitter(cold_frac=0.2, val_frac=0.1).split(medium_dataset, seed=1)
    assert res.cold_in_train() == set()
    train_items = set(res.train[C.Item].unique())
    assert set(res.cold_items).isdisjoint(train_items)


def test_val_items_not_in_train(medium_dataset: Dataset) -> None:
    res = PseudoColdSplitter(cold_frac=0.2, val_frac=0.1).split(medium_dataset, seed=1)
    train_items = set(res.train[C.Item].unique())
    val_items = set(res.val[C.Item].unique())
    assert val_items.isdisjoint(train_items)
    # val и test тоже не пересекаются по айтемам
    assert val_items.isdisjoint(set(res.cold_items))


def test_cold_interactions_only_in_test(medium_dataset: Dataset) -> None:
    res = PseudoColdSplitter(cold_frac=0.2, val_frac=0.1).split(medium_dataset, seed=1)
    cold = set(res.cold_items)
    # все взаимодействия cold-айтемов присутствуют в test и нигде больше
    assert set(res.test[C.Item].unique()) == cold
    assert cold.isdisjoint(set(res.train[C.Item].unique()))
    assert cold.isdisjoint(set(res.val[C.Item].unique()))


def test_warm_items_are_train_items(medium_dataset: Dataset) -> None:
    res = PseudoColdSplitter(cold_frac=0.2, val_frac=0.1).split(medium_dataset, seed=1)
    assert set(res.warm_items) == set(res.train[C.Item].unique())


def test_determinism_same_seed(medium_dataset: Dataset) -> None:
    a = PseudoColdSplitter(cold_frac=0.2).split(medium_dataset, seed=7)
    b = PseudoColdSplitter(cold_frac=0.2).split(medium_dataset, seed=7)
    assert np.array_equal(a.cold_items, b.cold_items)


def test_different_seed_changes_split(medium_dataset: Dataset) -> None:
    a = PseudoColdSplitter(cold_frac=0.2).split(medium_dataset, seed=1)
    b = PseudoColdSplitter(cold_frac=0.2).split(medium_dataset, seed=2)
    assert not np.array_equal(a.cold_items, b.cold_items)


def test_cold_fraction_approximately(medium_dataset: Dataset) -> None:
    res = PseudoColdSplitter(cold_frac=0.25, val_frac=0.0).split(medium_dataset, seed=3)
    n_items = medium_dataset.interactions[C.Item].nunique()
    frac = len(res.cold_items) / n_items
    assert 0.15 <= frac <= 0.35  # стратификация по бакетам даёт небольшой разброс


def test_cold_items_have_ground_truth(medium_dataset: Dataset) -> None:
    res = PseudoColdSplitter(cold_frac=0.2, min_item_interactions=2).split(medium_dataset, seed=1)
    counts = res.test.groupby(C.Item).size()
    assert (counts >= 2).all()


def test_small_dataset_still_gets_cold() -> None:
    # на малом датасете независимое округление по бакетам дало бы 0 cold;
    # глобальный target гарантирует ≥1
    rng = np.random.default_rng(0)
    users, items = [], []
    for item in range(6):
        for u in range(rng.integers(2, 6)):
            users.append(u)
            items.append(item)
    inter = pd.DataFrame({C.User: users, C.Item: items, C.Weight: 1.0, C.Datetime: 0})
    ds = Dataset(interactions=inter, name="small")
    res = PseudoColdSplitter(cold_frac=0.2, val_frac=0.0).split(ds, seed=1)
    assert len(res.cold_items) >= 1
    assert res.cold_in_train() == set()


def test_cold_fraction_global_target() -> None:
    # суммарная доля cold соответствует глобальному round(total*frac)
    rng = np.random.default_rng(0)
    users, items = [], []
    for item in range(50):
        for u in range(rng.integers(3, 10)):
            users.append(int(u))
            items.append(item)
    inter = pd.DataFrame({C.User: users, C.Item: items, C.Weight: 1.0, C.Datetime: 0})
    ds = Dataset(interactions=inter, name="d50")
    res = PseudoColdSplitter(cold_frac=0.2, val_frac=0.1).split(ds, seed=1)
    assert len(res.cold_items) == round(50 * 0.2)


def test_invalid_fractions() -> None:
    with pytest.raises(ValueError, match="cold_frac"):
        PseudoColdSplitter(cold_frac=0.0)
    with pytest.raises(ValueError, match="< 1"):
        PseudoColdSplitter(cold_frac=0.6, val_frac=0.5)
