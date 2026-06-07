"""High-level entry point: auto-select a cold-start method on the user's own data.

``recommend()`` runs every feasible method on an honest pseudo-cold holdout of the user's
warm items, ranks them by a headline metric, and returns an ``AutoResult`` with a
leaderboard, a verdict, and a ready-to-use fitted winner. See the design doc.
"""

from __future__ import annotations

from warmtransfer.methods.base import INPUT_KINDS, methods

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
