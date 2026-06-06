"""Known-answer тесты ранжирующих метрик (эталоны посчитаны вручную).

Сценарий: ranked = [A, B, C, D] = [1, 2, 3, 4], relevant = {2, 4}.
"""

from __future__ import annotations

import math

import pytest

from coldscore.metrics.ranking import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

RANKED = [1, 2, 3, 4]
RELEVANT = {2, 4}


def test_recall() -> None:
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0          # top1={1}
    assert recall_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.5)  # top2={1,2}, hit {2}
    assert recall_at_k(RANKED, RELEVANT, 4) == pytest.approx(1.0)


def test_precision() -> None:
    assert precision_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.5)   # 1/2
    assert precision_at_k(RANKED, RELEVANT, 4) == pytest.approx(0.5)   # 2/4


def test_average_precision() -> None:
    # hit на pos2 → 1/2; hit на pos4 → 2/4; sum=1.0; denom=min(2,4)=2
    assert average_precision_at_k(RANKED, RELEVANT, 4) == pytest.approx(0.5)


def test_ndcg() -> None:
    dcg = 1 / math.log2(3) + 1 / math.log2(5)   # B на pos2, D на pos4
    idcg = 1 / math.log2(2) + 1 / math.log2(3)  # 2 релевантных на идеальных позициях
    assert ndcg_at_k(RANKED, RELEVANT, 4) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(RANKED, RELEVANT, 1) == 0.0


def test_mrr() -> None:
    assert reciprocal_rank_at_k(RANKED, RELEVANT, 4) == pytest.approx(0.5)  # первый на pos2
    assert reciprocal_rank_at_k([1, 3, 5], RELEVANT, 3) == 0.0  # релевантных в топе нет


def test_empty_relevant_returns_nan() -> None:
    funcs = (
        recall_at_k,
        precision_at_k,
        average_precision_at_k,
        ndcg_at_k,
        reciprocal_rank_at_k,
    )
    for func in funcs:
        assert math.isnan(func(RANKED, set(), 4))


def test_k_larger_than_list() -> None:
    # k больше длины списка — без падения
    assert recall_at_k([1, 2], {2}, 10) == pytest.approx(1.0)
    assert ndcg_at_k([1, 2], {2}, 10) == pytest.approx(1 / math.log2(3))


def test_duplicates_in_ranked_do_not_exceed_one() -> None:
    # повтор одного relevant-айтема не должен давать метрику > 1 (робастность)
    assert average_precision_at_k([1, 1], {1}, 2) <= 1.0
    assert ndcg_at_k([1, 1], {1}, 2) <= 1.0
    # первый вход засчитан (pos1), повтор проигнорирован
    assert average_precision_at_k([1, 1], {1}, 2) == pytest.approx(1.0)
    assert ndcg_at_k([1, 1], {1}, 2) == pytest.approx(1.0)
