"""Recommendation quality metrics (our own correct implementation).

Public entry point — :func:`calc_metrics`: takes recommendations and ground truth in
``Columns`` format and returns a dict ``{"recall@10": ..., "auc": ...}``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.metrics._topk import ranked_lists, relevant_sets
from warmtransfer.metrics.classification import auc
from warmtransfer.metrics.ranking import RANKING_FUNCS, mean_over_users
from warmtransfer.metrics.relative import rela_impr

__all__ = [
    "auc",
    "calc_metrics",
    "global_auc",
    "mean_user_auc",
    "ranking_metrics",
    "rela_impr",
]


def ranking_metrics(
    reco: pd.DataFrame,
    ground_truth: pd.DataFrame,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """All ranking metrics for all ``k`` values: ``{"recall@10": ..., ...}``."""
    ranked = ranked_lists(reco)
    relevant = relevant_sets(ground_truth)
    out: dict[str, float] = {}
    for name, func in RANKING_FUNCS.items():
        for k in ks:
            out[f"{name}@{k}"] = mean_over_users(ranked, relevant, func, k)
    return out


def mean_user_auc(reco: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """Mean per-user AUC.

    ``reco`` must contain scores for ALL candidate items of each user
    (the full user × cold_item grid). Positives are items from ``ground_truth``.
    Averaged over users that have both classes present.
    """
    relevant = relevant_sets(ground_truth)
    vals: list[float] = []
    for user, grp in reco.groupby(C.User, sort=False):
        rel = relevant.get(user)
        if not rel:
            continue
        items = grp[C.Item].to_numpy()
        scores = grp[C.Score].to_numpy()
        y_true = np.fromiter((1 if it in rel else 0 for it in items), dtype=int, count=len(items))
        val = auc(y_true, scores)
        if not math.isnan(val):
            vals.append(val)
    return float(np.mean(vals)) if vals else math.nan


def global_auc(reco: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """Global AUC: pool all (user, cold_item) pairs with 1/0 labels into one common AUC.

    Exposes the popularity signal more strongly than per-user averaging; used to
    cross-check against protocols where AUC is computed over all pairs at once.
    """
    gt_pairs = set(zip(ground_truth[C.User], ground_truth[C.Item], strict=True))
    pairs = list(zip(reco[C.User], reco[C.Item], strict=True))
    y_true = np.fromiter((1 if p in gt_pairs else 0 for p in pairs), dtype=int, count=len(pairs))
    return auc(y_true, reco[C.Score].to_numpy())


def calc_metrics(
    reco: pd.DataFrame,
    ground_truth: pd.DataFrame,
    ks: tuple[int, ...] = (1, 5, 10),
    *,
    include_auc: bool = True,
) -> dict[str, float]:
    """Full metric set: ranking@k (+ per-user and global AUC over the candidate grid)."""
    out = ranking_metrics(reco, ground_truth, ks)
    if include_auc:
        out["auc"] = mean_user_auc(reco, ground_truth)
        out["auc_global"] = global_auc(reco, ground_truth)
    return out
