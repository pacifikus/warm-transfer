"""Tests for the requires-driven feasibility filter."""

from __future__ import annotations

from warmtransfer.recommend import (
    EMBEDDING_METHODS,
    available_inputs_for,
    feasible_methods,
)


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
