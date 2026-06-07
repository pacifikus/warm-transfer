"""Метрики качества рекомендаций (собственная корректная реализация).

Публичный вход — :func:`calc_metrics`: принимает рекомендации и ground truth в
формате ``Columns`` и возвращает словарь ``{"recall@10": ..., "auc": ...}``.
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
    """Все ранжирующие метрики для всех ``k``: ``{"recall@10": ..., ...}``."""
    ranked = ranked_lists(reco)
    relevant = relevant_sets(ground_truth)
    out: dict[str, float] = {}
    for name, func in RANKING_FUNCS.items():
        for k in ks:
            out[f"{name}@{k}"] = mean_over_users(ranked, relevant, func, k)
    return out


def mean_user_auc(reco: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """Средний по пользователям AUC.

    ``reco`` должен содержать скоры по ВСЕМ айтемам-кандидатам каждого пользователя
    (полная сетка user × cold_item). Положительные — айтемы из ``ground_truth``.
    Усредняется по пользователям, у которых присутствуют оба класса.
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
    """Глобальный AUC: пул всех пар (user, cold_item) с метками 1/0 — один общий AUC.

    Сильнее проявляет popularity-сигнал, чем per-user усреднение; используется для
    сверки с протоколами, где AUC считается по всем парам сразу.
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
    """Полный набор метрик: ranking@k (+ per-user и глобальный AUC по сетке кандидатов)."""
    out = ranking_metrics(reco, ground_truth, ks)
    if include_auc:
        out["auc"] = mean_user_auc(reco, ground_truth)
        out["auc_global"] = global_auc(reco, ground_truth)
    return out
