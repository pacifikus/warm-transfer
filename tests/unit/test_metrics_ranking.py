"""Known-answer tests for ranking metrics (reference values computed by hand).

Scenario: ranked = [A, B, C, D] = [1, 2, 3, 4], relevant = {2, 4}.
"""

from __future__ import annotations

import math

import pytest

from warmtransfer.metrics.ranking import (
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
    assert recall_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.5)  # top2={1,2}, hits {2}
    assert recall_at_k(RANKED, RELEVANT, 4) == pytest.approx(1.0)


def test_precision() -> None:
    assert precision_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.5)   # 1/2
    assert precision_at_k(RANKED, RELEVANT, 4) == pytest.approx(0.5)   # 2/4


def test_average_precision() -> None:
    # hit at pos2 -> 1/2; hit at pos4 -> 2/4; sum=1.0; denom=min(2,4)=2
    assert average_precision_at_k(RANKED, RELEVANT, 4) == pytest.approx(0.5)


def test_ndcg() -> None:
    dcg = 1 / math.log2(3) + 1 / math.log2(5)   # B at pos2, D at pos4
    idcg = 1 / math.log2(2) + 1 / math.log2(3)  # 2 relevant items at ideal positions
    assert ndcg_at_k(RANKED, RELEVANT, 4) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(RANKED, RELEVANT, 1) == 0.0


def test_mrr() -> None:
    assert reciprocal_rank_at_k(RANKED, RELEVANT, 4) == pytest.approx(0.5)  # first at pos2
    assert reciprocal_rank_at_k([1, 3, 5], RELEVANT, 3) == 0.0  # no relevant items in top


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
    # k larger than the list length -- must not crash
    assert recall_at_k([1, 2], {2}, 10) == pytest.approx(1.0)
    assert ndcg_at_k([1, 2], {2}, 10) == pytest.approx(1 / math.log2(3))


def test_duplicates_in_ranked_do_not_exceed_one() -> None:
    # a repeated relevant item must not push the metric above 1 (robustness)
    assert average_precision_at_k([1, 1], {1}, 2) <= 1.0
    assert ndcg_at_k([1, 1], {1}, 2) <= 1.0
    # the first occurrence is counted (pos1), the repeat is ignored
    assert average_precision_at_k([1, 1], {1}, 2) == pytest.approx(1.0)
    assert ndcg_at_k([1, 1], {1}, 2) == pytest.approx(1.0)
