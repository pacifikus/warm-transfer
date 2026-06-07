"""Property-based metric tests (hypothesis): invariants that must always hold."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from warmtransfer.metrics.ranking import ndcg_at_k, precision_at_k, recall_at_k

# lists of unique item_id and a subset of relevant ones
items_strategy = st.lists(
    st.integers(min_value=0, max_value=50), min_size=1, max_size=15, unique=True
)


@given(items=items_strategy, data=st.data())
def test_recall_in_unit_interval(items: list[int], data: st.DataObject) -> None:
    relevant = set(data.draw(st.lists(st.sampled_from(items), min_size=1, unique=True)))
    k = data.draw(st.integers(min_value=1, max_value=len(items) + 5))
    assert 0.0 <= recall_at_k(items, relevant, k) <= 1.0


@given(items=items_strategy, data=st.data())
def test_ndcg_in_unit_interval(items: list[int], data: st.DataObject) -> None:
    relevant = set(data.draw(st.lists(st.sampled_from(items), min_size=1, unique=True)))
    k = data.draw(st.integers(min_value=1, max_value=len(items) + 5))
    assert 0.0 <= ndcg_at_k(items, relevant, k) <= 1.0 + 1e-9


@given(items=items_strategy, data=st.data())
def test_precision_at_most_one(items: list[int], data: st.DataObject) -> None:
    relevant = set(data.draw(st.lists(st.sampled_from(items), min_size=1, unique=True)))
    k = data.draw(st.integers(min_value=1, max_value=len(items) + 5))
    assert precision_at_k(items, relevant, k) <= 1.0 + 1e-9


@given(items=items_strategy, data=st.data())
def test_recall_monotonic_in_k(items: list[int], data: st.DataObject) -> None:
    relevant = set(data.draw(st.lists(st.sampled_from(items), min_size=1, unique=True)))
    k = data.draw(st.integers(min_value=1, max_value=len(items)))
    # recall@k is non-decreasing in k
    assert recall_at_k(items, relevant, k) <= recall_at_k(items, relevant, k + 1) + 1e-9
