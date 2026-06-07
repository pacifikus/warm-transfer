"""Ranking metrics (per-user). Binary relevance.

All functions take:
  * ``ranked`` — sequence of item_id in ranking order (best first);
  * ``relevant`` — set of relevant item_id;
  * ``k`` — cutoff.

Convention: if a user has no relevant items, the metric is undefined and
``nan`` is returned (such users are excluded from averaging).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _topk(ranked: Sequence, k: int) -> list:
    return list(ranked[:k])


def recall_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    if not relevant:
        return math.nan
    hits = len(set(_topk(ranked, k)) & relevant)
    return hits / len(relevant)


def precision_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    if not relevant:
        return math.nan
    hits = len(set(_topk(ranked, k)) & relevant)
    return hits / k


def average_precision_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    """AP@k = (Σ P@i · rel(i)) / min(|relevant|, k).

    Duplicates of the same relevant item in ranked are counted once
    (otherwise the metric could exceed 1).
    """
    if not relevant:
        return math.nan
    score = 0.0
    hits = 0
    seen: set = set()
    for i, item in enumerate(_topk(ranked, k), start=1):
        if item in relevant and item not in seen:
            seen.add(item)
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)


def ndcg_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    """NDCG@k with binary relevance: DCG@k / IDCG@k.

    Each relevant item is counted once (a repeat yields no extra gain).
    """
    if not relevant:
        return math.nan
    dcg = 0.0
    seen: set = set()
    for i, item in enumerate(_topk(ranked, k), start=1):
        if item in relevant and item not in seen:
            seen.add(item)
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg


def reciprocal_rank_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    """RR@k = 1 / (position of the first relevant item), 0 if none in top-k."""
    if not relevant:
        return math.nan
    for i, item in enumerate(_topk(ranked, k), start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


#: Registry of per-user ranking metrics: name -> function (ranked, relevant, k) -> float.
RANKING_FUNCS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "map": average_precision_at_k,
    "ndcg": ndcg_at_k,
    "mrr": reciprocal_rank_at_k,
}


def mean_over_users(
    ranked_by_user: dict,
    relevant_by_user: dict,
    func,
    k: int,
) -> float:
    """Average a per-user metric over users that have relevant items."""
    vals = []
    for user, relevant in relevant_by_user.items():
        if not relevant:
            continue
        ranked = ranked_by_user.get(user, [])
        vals.append(func(ranked, relevant, k))
    return float(np.mean(vals)) if vals else math.nan
