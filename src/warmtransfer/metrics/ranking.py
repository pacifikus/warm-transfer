"""Ранжирующие метрики (per-user). Бинарная релевантность.

Все функции принимают:
  * ``ranked`` — последовательность item_id в порядке выдачи (лучший первым);
  * ``relevant`` — множество релевантных item_id;
  * ``k`` — отсечка.

Соглашение: если у пользователя нет релевантных айтемов, метрика не определена и
возвращается ``nan`` (такие пользователи исключаются при усреднении).
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

    Дубликаты одного и того же relevant-айтема в ranked засчитываются один раз
    (иначе метрика может превысить 1).
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
    """NDCG@k при бинарной релевантности: DCG@k / IDCG@k.

    Каждый relevant-айтем учитывается один раз (повтор не даёт лишнего gain).
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
    """RR@k = 1 / (позиция первого релевантного), 0 если его нет в топ-k."""
    if not relevant:
        return math.nan
    for i, item in enumerate(_topk(ranked, k), start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


#: Реестр per-user ранжирующих метрик: имя -> функция (ranked, relevant, k) -> float.
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
    """Усреднить per-user метрику по пользователям, у которых есть релевантные айтемы."""
    vals = []
    for user, relevant in relevant_by_user.items():
        if not relevant:
            continue
        ranked = ranked_by_user.get(user, [])
        vals.append(func(ranked, relevant, k))
    return float(np.mean(vals)) if vals else math.nan
