"""Построение ранжированных списков с детерминированным tie-breaking.

Соглашение (фиксируем своё): при равных скорах порядок задаётся возрастанием
``item_id`` — это делает метрики воспроизводимыми независимо от порядка строк во входе.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C


def rank_items(items: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Отсортировать ``items`` по убыванию ``scores``; ties → по возрастанию item_id.

    Возвращает массив item_id в порядке выдачи (лучший первым).
    """
    items = np.asarray(items)
    scores = np.asarray(scores, dtype=float)
    # lexsort: последний ключ — старший. Старший = -score (убывание), младший = item (возрастание).
    order = np.lexsort((items, -scores))
    return items[order]


def ranked_lists(reco: pd.DataFrame) -> dict:
    """Из DataFrame ``[user_id, item_id, score]`` собрать ``{user_id: [items по убыв. score]}``."""
    out: dict = {}
    for user, grp in reco.groupby(C.User, sort=False):
        out[user] = rank_items(grp[C.Item].to_numpy(), grp[C.Score].to_numpy())
    return out


def relevant_sets(ground_truth: pd.DataFrame) -> dict[object, set]:
    """Из DataFrame взаимодействий собрать ``{user_id: {релевантные item_id}}``."""
    out: dict[object, set] = {}
    for user, grp in ground_truth.groupby(C.User, sort=False):
        out[user] = set(grp[C.Item].tolist())
    return out
