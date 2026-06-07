"""High-level entry point: auto-select a cold-start method on the user's own data.

``recommend()`` runs every feasible method on an honest pseudo-cold holdout of the user's
warm items, ranks them by a headline metric, and returns an ``AutoResult`` with a
leaderboard, a verdict, and a ready-to-use fitted winner. See the design doc.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.holdout import HoldoutConfig, pseudo_cold_split
from warmtransfer.methods.base import INPUT_KINDS, methods
from warmtransfer.similarity import content_similarity
from warmtransfer.types import Dataset, ItemFeatures, SplitResult, TransferInputs

#: Trivial reference methods (not "transfer"); used by recommend()'s verdict step as the baseline.
BASELINE_METHODS = frozenset(
    {"random", "most_popular", "grouped_most_popular", "grouped_most_popular_pers"}
)
#: Embedding-family methods — out of scope for recommend v1 (need donor embeddings).
EMBEDDING_METHODS = frozenset(
    {"embedding_avg", "attention_emb", "linmap_emb", "dropoutnet", "magnitude_scaling"}
)


def feasible_methods(
    present: set[str], requested: list[str] | None
) -> tuple[list[str], dict[str, str]]:
    """Split registered methods into runnable vs skipped given the available inputs.

    ``present`` is the set of available input kinds (subset of ``INPUT_KINDS``). ``requested``
    optionally restricts to a subset of method names. Returns ``(run, skipped)`` where
    ``skipped`` maps a method name to a human-readable reason.
    """
    all_names = list(methods.names())
    candidates = all_names if requested is None else [m for m in requested if m in all_names]

    run: list[str] = []
    skipped: dict[str, str] = {}
    for name in candidates:
        if name in EMBEDDING_METHODS:
            skipped[name] = "embedding methods are out of scope for recommend v1"
            continue
        req = methods.get(name).requires
        missing = req - present
        if missing:
            skipped[name] = f"missing inputs: {sorted(missing)}"
            continue
        run.append(name)

    # methods the user explicitly named but that don't exist
    if requested is not None:
        for name in requested:
            if name not in all_names:
                skipped[name] = "unknown method"
    return run, skipped


def available_inputs_for(
    *, has_scores: bool, has_content: bool, has_train: bool, has_val: bool, has_meta: bool
) -> set[str]:
    """Map high-level input flags to the ``INPUT_KINDS`` present set (no embeddings in v1)."""
    present: set[str] = set()
    if has_scores:
        present.add("donor_scores")
    if has_content:
        present.add("content")
        # recommend always derives similarity from content (content_similarity),
        # so content availability implies similarity availability
        present.add("similarity")
    if has_train:
        present.add("train_interactions")
    if has_val:
        present.add("val")
    if has_meta:
        present.add("item_meta")
    assert present <= INPUT_KINDS
    return present


def _scores_for(donor_scores: pd.DataFrame, item_ids: np.ndarray) -> pd.DataFrame:
    """Filter donor_scores to the given item_ids (index reset)."""
    mask = donor_scores[C.Item].isin(item_ids.tolist())
    return donor_scores.loc[mask].reset_index(drop=True)


def build_holdout_inputs(
    split: SplitResult,
    content: ItemFeatures,
    donor_scores: pd.DataFrame,
    item_meta: pd.DataFrame | None,
) -> TransferInputs:
    """Assemble ``TransferInputs`` for evaluating methods on one holdout fold.

    Donor scores are restricted to warm items (the cold/val-cold items are "unseen").
    Similarity is computed from content; the val-cold fold is attached when present.
    """
    warm_ids = split.warm_items
    test_cold_ids = split.cold_items
    warm_feat = content.subset(warm_ids)
    cold_feat = content.subset(test_cold_ids)

    inputs = TransferInputs(
        donor_scores=_scores_for(donor_scores, warm_ids),
        train_interactions=split.train,
        warm_features=warm_feat,
        cold_features=cold_feat,
        similarity=content_similarity(cold_feat, warm_feat),
        warm_items=warm_ids,
        cold_items=test_cold_ids,
        item_meta=item_meta,
    )

    if len(split.val):
        val_cold_ids = np.asarray(pd.unique(split.val[C.Item]))
        val_feat = content.subset(val_cold_ids)
        inputs.val_cold_features = val_feat
        inputs.val_similarity = content_similarity(val_feat, warm_feat)
        inputs.val_interactions = split.val
    return inputs


def build_production_inputs(
    interactions: pd.DataFrame,
    content: ItemFeatures,
    donor_scores: pd.DataFrame,
    cold_item_ids: np.ndarray,
    item_meta: pd.DataFrame | None,
    *,
    needs_val: bool,
    holdout: HoldoutConfig,
    seed: int,
) -> TransferInputs:
    """Assemble ``TransferInputs`` to refit the winner on ALL warm data for real cold items.

    Warm = all items present in ``donor_scores``. For val-family winners we carve an internal
    val-cold fold out of warm via the same pseudo-cold split (the real cold items stay the
    prediction target).
    """
    warm_ids = np.asarray(pd.unique(donor_scores[C.Item]))
    warm_feat = content.subset(warm_ids)
    cold_ids = np.asarray(cold_item_ids)
    cold_feat = content.subset(cold_ids)

    inputs = TransferInputs(
        donor_scores=_scores_for(donor_scores, warm_ids),
        train_interactions=interactions,
        warm_features=warm_feat,
        cold_features=cold_feat,
        similarity=content_similarity(cold_feat, warm_feat),
        warm_items=warm_ids,
        cold_items=cold_ids,
        item_meta=item_meta,
    )

    if needs_val:
        warm_inter = interactions.loc[interactions[C.Item].isin(warm_ids.tolist())]
        sub = pseudo_cold_split(Dataset(warm_inter), holdout, seed)
        val_ids = np.asarray(pd.unique(sub.val[C.Item])) if len(sub.val) else np.asarray([])
        if len(val_ids):
            carved_warm_feat = content.subset(sub.warm_items)
            val_feat = content.subset(val_ids)
            # narrow warm to sub.warm_items (exclude carved val-cold); cold_feat/cold_items
            # stay unchanged — the real cold items remain the prediction target
            inputs.donor_scores = _scores_for(donor_scores, sub.warm_items)
            inputs.warm_features = carved_warm_feat
            inputs.warm_items = sub.warm_items
            inputs.similarity = content_similarity(cold_feat, carved_warm_feat)
            inputs.val_cold_features = val_feat
            inputs.val_similarity = content_similarity(val_feat, carved_warm_feat)
            inputs.val_interactions = sub.val
        else:
            warnings.warn(
                "needs_val=True, but the internal holdout produced no val-cold items "
                "(dataset too small or min_item_interactions too high); the winner is "
                "refit without a val fold and may degrade.",
                stacklevel=2,
            )
    return inputs
