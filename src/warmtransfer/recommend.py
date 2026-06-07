"""High-level entry point: auto-select a cold-start method on the user's own data.

``recommend()`` runs every feasible method on an honest pseudo-cold holdout of the user's
warm items, ranks them by a headline metric, and returns an ``AutoResult`` with a
leaderboard, a verdict, and a ready-to-use fitted winner. See the design doc.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.holdout import HoldoutConfig, pseudo_cold_split
from warmtransfer.methods.base import INPUT_KINDS, ColdStartMethod, methods
from warmtransfer.metrics import calc_metrics
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


@dataclass
class AutoResult:
    """Outcome of :func:`recommend`: leaderboard + verdict + a ready-to-use winner."""

    leaderboard: pd.DataFrame  # index = method, columns = metrics
    best: str
    best_transfer: str | None
    baseline_best: str | None
    metric: str
    verdict: str
    skipped: dict[str, str] = field(default_factory=dict)
    _ctx: dict = field(default_factory=dict, repr=False)
    _fitted: ColdStartMethod | None = field(default=None, repr=False)
    _fitted_key: tuple | None = field(default=None, repr=False)

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        """Predict scores for new cold items with the winner refit on ALL warm data.

        The winner is refit lazily; the fit is cached per distinct ``cold_item_ids`` set
        (a different set triggers a refit, since the production inputs depend on it).
        """
        cold = np.asarray(cold_item_ids)
        key = tuple(cold.tolist())
        winner = self.best_transfer or self.best
        if self._fitted is None or self._fitted_key != key:
            ctx = self._ctx
            needs_val = "val" in methods.get(winner).requires
            inputs = build_production_inputs(
                ctx["interactions"], ctx["content"], ctx["donor_scores"],
                cold, ctx["item_meta"],
                needs_val=needs_val, holdout=ctx["holdout"], seed=ctx["seed"],
            )
            self._fitted = methods.get(winner)().fit(inputs, ctx["seed"])
            self._fitted_key = key
        return self._fitted.predict(np.asarray(user_ids), cold)

    def __str__(self) -> str:
        cols = [self.metric] + [c for c in self.leaderboard.columns if c != self.metric]
        table = self.leaderboard[cols].round(4).to_string()
        lines = [f"Headline metric: {self.metric}", table, "", "Verdict:", self.verdict]
        if self.skipped:
            lines += ["", "Skipped:"] + [f"  - {k}: {v}" for k, v in self.skipped.items()]
        return "\n".join(lines)


def _make_verdict(
    board: pd.DataFrame, metric: str, best_transfer: str | None,
    baseline_best: str | None, min_rela_gain: float,
) -> str:
    caveat = (
        "Estimated on a holdout of your warm items; the donor was not retrained, "
        "so treat this as a mildly optimistic estimate."
    )
    if best_transfer is None:
        return "No transfer method was feasible (only baselines available). " + caveat
    t = float(board.loc[best_transfer, metric])
    if baseline_best is None:
        return f"Best method: {best_transfer} ({metric}={t:.4f}). {caveat}"
    b = float(board.loc[baseline_best, metric])
    gain = float("inf") if b <= 0 else (t - b) / abs(b)
    if gain >= min_rela_gain:
        return (
            f"Best method: {best_transfer} ({metric}={t:.4f}), "
            f"{gain * 100:+.1f}% over baseline {baseline_best} ({b:.4f}). "
            f"Transfer works on your data - worth using. {caveat}"
        )
    relation = "underperforms" if gain < 0 else "barely beats"
    return (
        f"Best transfer method {best_transfer} ({metric}={t:.4f}) {relation} "
        f"baseline {baseline_best} ({b:.4f}) ({gain * 100:+.1f}%). "
        f"Transfer adds little on this data. {caveat}"
    )


def _eval_method(
    name: str, inputs: TransferInputs, split: SplitResult, ks: tuple[int, ...], seed: int
) -> dict[str, float]:
    """Fit one method on the holdout and score it on test-cold. Raises on failure."""
    eval_users = np.asarray(pd.unique(split.test[C.User]))
    model = methods.get(name)().fit(inputs, seed)
    reco = model.predict(eval_users, split.cold_items)
    return calc_metrics(reco, split.test, ks)


def recommend(
    interactions: pd.DataFrame,
    content: ItemFeatures,
    donor_scores: pd.DataFrame,
    *,
    item_meta: pd.DataFrame | None = None,
    metric: str = "auc",
    ks: tuple[int, ...] = (1, 5, 10),
    methods: list[str] | None = None,
    holdout: HoldoutConfig | None = None,
    seed: int = 42,
    n_seeds: int = 1,
    min_rela_gain: float = 0.02,
    verbose: bool = True,
) -> AutoResult:
    """Auto-select a cold-start method on the user's own data via an honest holdout.

    ``metric`` is the headline metric (default per-user ``auc``). ``methods`` optionally
    restricts the candidate method names. Embedding methods are skipped in v1. Returns an
    :class:`AutoResult` with a leaderboard, a verdict, and a ready-to-use fitted winner.
    """
    if interactions is None or len(interactions) == 0:
        raise ValueError(
            "recommend() needs interactions to evaluate methods honestly. "
            "Provide a [user_id, item_id] history of your warm items."
        )
    cfg = holdout or HoldoutConfig()
    present = available_inputs_for(
        has_scores=len(donor_scores) > 0, has_content=True,
        has_train=len(interactions) > 0, has_val=cfg.val_frac > 0,
        has_meta=item_meta is not None,
    )
    run, skipped = feasible_methods(present, methods)
    if not run:
        raise ValueError(f"No feasible methods for the given inputs. Skipped: {skipped}")

    seeds = [seed + i for i in range(max(1, n_seeds))]
    per_seed: list[pd.DataFrame] = []
    for s in seeds:
        split = pseudo_cold_split(Dataset(interactions, content, "user-data"), cfg, s)
        inputs = build_holdout_inputs(split, content, donor_scores, item_meta)
        rows: dict[str, dict[str, float]] = {}
        for name in run:
            try:
                rows[name] = _eval_method(name, inputs, split, ks, s)
            except Exception as exc:
                skipped[name] = f"failed: {type(exc).__name__}: {exc}"
                if verbose:
                    print(f"[recommend] skip {name}: {exc}")
        per_seed.append(pd.DataFrame.from_dict(rows, orient="index"))

    board: pd.DataFrame = pd.concat(per_seed).groupby(level=0).mean()  # type: ignore[assignment]
    if board.empty or metric not in board.columns:
        raise RuntimeError(
            f"All methods failed during evaluation; nothing to rank. Skipped: {skipped}"
        )
    board = board.sort_values(metric, ascending=False)

    ran = [m for m in board.index if m not in skipped]
    if not ran:
        raise RuntimeError(f"No method produced a usable result. Skipped: {skipped}")
    transfers = [m for m in ran if m not in BASELINE_METHODS]
    baselines = [m for m in ran if m in BASELINE_METHODS]
    best_transfer = transfers[0] if transfers else None
    baseline_best = baselines[0] if baselines else None
    best = ran[0]

    verdict = _make_verdict(board, metric, best_transfer, baseline_best, min_rela_gain)
    result = AutoResult(
        leaderboard=board, best=best, best_transfer=best_transfer,
        baseline_best=baseline_best, metric=metric, verdict=verdict, skipped=skipped,
        _ctx={"interactions": interactions, "content": content, "donor_scores": donor_scores,
              "item_meta": item_meta, "holdout": cfg, "seed": seed},
    )
    if verbose:
        print(result)
    return result
