"""Tests for the requires-driven feasibility filter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.holdout import HoldoutConfig, pseudo_cold_split
from warmtransfer.recommend import (
    EMBEDDING_METHODS,
    available_inputs_for,
    build_holdout_inputs,
    build_production_inputs,
    feasible_methods,
)
from warmtransfer.types import Dataset, ItemFeatures


def test_scores_plus_content_runs_transfer_and_baselines() -> None:
    present = {"donor_scores", "content", "similarity", "train_interactions"}
    run, skipped = feasible_methods(present, requested=None)
    assert "linmap" in run and "knn_score_avg" in run
    assert "most_popular" in run and "grouped_most_popular_pers" in run
    # val-only methods skipped (no val fold available yet)
    assert "stacking_plus" in skipped
    assert "stacking" in skipped
    # embedding methods always skipped in v1
    for m in EMBEDDING_METHODS:
        assert m in skipped


def test_val_methods_run_when_val_present() -> None:
    present = {"donor_scores", "content", "similarity", "train_interactions", "val"}
    run, _skipped = feasible_methods(present, requested=None)
    assert "stacking_plus" in run and "stacking" in run and "logreg_calib" in run


def test_requested_subset_restricts() -> None:
    present = {"donor_scores", "content"}
    run, _skipped = feasible_methods(present, requested=["linmap"])
    assert run == ["linmap"]
    assert "knn_score_avg" not in run


def test_requested_unfeasible_goes_to_skipped() -> None:
    present = {"donor_scores", "content"}  # no train_interactions
    run, skipped = feasible_methods(present, requested=["grouped_most_popular"])
    assert "grouped_most_popular" in skipped
    assert run == []


def test_requested_unknown_method() -> None:
    present = {"donor_scores", "content", "similarity"}
    run, skipped = feasible_methods(present, requested=["linmap", "nonexistent_method"])
    assert run == ["linmap"]
    assert skipped.get("nonexistent_method") == "unknown method"


def test_content_implies_similarity() -> None:
    present = available_inputs_for(
        has_scores=True, has_content=True, has_train=True, has_val=False, has_meta=False
    )
    assert "similarity" in present and "content" in present
    # similarity-dependent transfer methods become feasible without an explicit similarity input
    run, _ = feasible_methods(present, requested=None)
    assert "knn_score_avg" in run and "attention_knn" in run


def _content(n_items: int = 20) -> ItemFeatures:
    rng = np.random.default_rng(1)
    return ItemFeatures(
        item_ids=np.arange(n_items),
        matrix=rng.random((n_items, 4)),
        feature_names=[f"f{i}" for i in range(4)],
    )


def _interactions(n_users: int = 30, n_items: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = []
    for it in range(n_items):
        for u in rng.choice(n_users, size=3 + it % 5, replace=False):
            rows.append((int(u), it))
    return pd.DataFrame(rows, columns=[C.User, C.Item])  # type: ignore[arg-type]


def test_build_holdout_inputs_strips_cold_scores() -> None:
    inter = _interactions()
    content = _content()
    pairs = inter[[C.User, C.Item]].drop_duplicates()
    donor = pairs.assign(**{C.Score: 1.0})
    split = pseudo_cold_split(Dataset(inter), HoldoutConfig(min_item_interactions=3), seed=5)

    inputs = build_holdout_inputs(split, content, donor, item_meta=None)
    assert set(inputs.donor_scores[C.Item]) & set(split.cold_items) == set()
    assert inputs.warm_features is not None
    assert inputs.cold_features is not None
    assert inputs.similarity is not None
    assert set(inputs.warm_features.item_ids) == set(split.warm_items)
    assert set(inputs.cold_features.item_ids) == set(split.cold_items)
    assert inputs.similarity.shape == (len(split.cold_items), len(split.warm_items))
    assert inputs.val_interactions is not None
    assert inputs.val_similarity is not None
    assert inputs.val_similarity.shape == (len(np.unique(split.val[C.Item])), len(split.warm_items))


def test_build_production_inputs_val_alignment() -> None:
    inter = _interactions()
    content = _content()
    pairs = inter[[C.User, C.Item]].drop_duplicates()
    donor = pairs.assign(**{C.Score: 1.0})
    cold = np.array([0, 1])  # real cold items (present in content)
    inputs = build_production_inputs(
        inter, content, donor, cold, item_meta=None,
        needs_val=True, holdout=HoldoutConfig(min_item_interactions=3), seed=5,
    )
    assert inputs.warm_features is not None
    assert inputs.similarity is not None
    assert inputs.warm_items is not None
    n_warm = inputs.warm_features.n_items
    # all similarity matrices aligned to the SAME warm set
    assert inputs.similarity.shape == (len(cold), n_warm)
    if inputs.val_similarity is not None:
        assert inputs.val_similarity.shape[1] == n_warm
    # production donor scores exclude carved val-cold items
    donor_items = set(inputs.donor_scores[C.Item])
    assert donor_items & set(inputs.warm_items) == donor_items
