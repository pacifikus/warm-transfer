"""Ranking tests with tie-breaking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.metrics._topk import rank_items, ranked_lists, relevant_sets


def test_rank_items_orders_by_score_desc() -> None:
    items = np.array([10, 20, 30])
    scores = np.array([0.1, 0.9, 0.5])
    assert rank_items(items, scores).tolist() == [20, 30, 10]


def test_rank_items_tie_break_by_item_asc() -> None:
    # ties at score=0.5 between 3 and 1 -> by ascending item_id; 2 with score 0.9 first
    items = np.array([3, 1, 2])
    scores = np.array([0.5, 0.5, 0.9])
    assert rank_items(items, scores).tolist() == [2, 1, 3]


def test_ranked_lists_and_relevant_sets() -> None:
    reco = pd.DataFrame(
        {
            C.User: [1, 1, 2],
            C.Item: [10, 11, 20],
            C.Score: [0.2, 0.8, 0.5],
        }
    )
    ranked = ranked_lists(reco)
    assert ranked[1].tolist() == [11, 10]
    assert ranked[2].tolist() == [20]

    gt = pd.DataFrame({C.User: [1, 2], C.Item: [11, 20]})
    rel = relevant_sets(gt)
    assert rel == {1: {11}, 2: {20}}
