"""Анализ результатов по бакетам популярности cold-айтемов (голова/хвост).

Декомпозирует recall@k по уровню популярности целевого cold-айтема: показывает, на каких
айтемах метод выигрывает — на массовых (голова) или нишевых (хвост). Это раскрывает природу
популярностного bias, обещанную в `docs/eval-protocol.md`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coldscore.columns import Columns as C


def _topk_sets(reco: pd.DataFrame, k: int) -> dict:
    """Для каждого пользователя — множество top-k айтемов по убыванию скора."""
    ordered = reco.sort_values(C.Score, ascending=False)
    topk = ordered.groupby(C.User).head(k)
    out: dict = {}
    for u, it in zip(topk[C.User].to_numpy(), topk[C.Item].to_numpy(), strict=True):
        out.setdefault(u, set()).add(it)
    return out


def recall_by_popularity_bucket(
    reco: pd.DataFrame,
    ground_truth: pd.DataFrame,
    item_popularity: dict,
    *,
    n_buckets: int = 4,
    k: int = 10,
) -> pd.DataFrame:
    """recall@k, разложенный по бакетам популярности cold-айтемов.

    :param reco: рекомендации ``[user_id, item_id, score]``.
    :param ground_truth: истинные взаимодействия cold-айтемов ``[user_id, item_id]``.
    :param item_popularity: словарь ``item_id -> число взаимодействий`` (для бакетирования).
    :param n_buckets: число бакетов популярности (квантили), от хвоста к голове.
    :param k: глубина top-k.
    :return: DataFrame ``[bucket, n_relevant, recall@k, mean_pop, pop_range]``.
    """
    items = np.array(sorted(item_popularity))
    pops = np.array([item_popularity[i] for i in items], dtype=float)

    # квантильные границы по популярности cold-айтемов
    edges = np.quantile(pops, np.linspace(0, 1, n_buckets + 1))
    edges[-1] = np.inf
    bucket_of = {
        it: int(np.searchsorted(edges[1:-1], p, side="right"))
        for it, p in zip(items, pops, strict=True)
    }

    topk = _topk_sets(reco, k)
    hits = np.zeros(n_buckets)
    total = np.zeros(n_buckets)
    gt_users = ground_truth[C.User].to_numpy()
    gt_items = ground_truth[C.Item].to_numpy()
    for u, it in zip(gt_users, gt_items, strict=True):
        b = bucket_of.get(it)
        if b is None:
            continue
        total[b] += 1.0
        if it in topk.get(u, set()):
            hits[b] += 1.0

    item_buckets = np.array([bucket_of[i] for i in items])
    rows = []
    for b in range(n_buckets):
        in_bucket = pops[item_buckets == b]
        rng_str = f"{in_bucket.min():.0f}–{in_bucket.max():.0f}" if in_bucket.size else ""
        rows.append(
            {
                "bucket": b,
                "n_relevant": int(total[b]),
                f"recall@{k}": hits[b] / total[b] if total[b] > 0 else np.nan,
                "mean_pop": float(in_bucket.mean()) if in_bucket.size else np.nan,
                "pop_range": rng_str,
            }
        )
    return pd.DataFrame(rows)
