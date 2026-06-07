"""Building ranked lists with deterministic tie-breaking.

Convention (our own, fixed here): on equal scores the order is set by ascending
``item_id`` — this makes metrics reproducible regardless of the row order in the input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C


def rank_items(items: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Sort ``items`` by descending ``scores``; ties → by ascending item_id.

    Returns an array of item_id in output order (best first).
    """
    items = np.asarray(items)
    scores = np.asarray(scores, dtype=float)
    # lexsort: the last key is the most significant. Primary = -score (descending),
    # secondary = item (ascending).
    order = np.lexsort((items, -scores))
    return items[order]


def ranked_lists(reco: pd.DataFrame) -> dict:
    """From a DataFrame ``[user_id, item_id, score]`` build
    ``{user_id: [items sorted by descending score]}``."""
    out: dict = {}
    for user, grp in reco.groupby(C.User, sort=False):
        out[user] = rank_items(grp[C.Item].to_numpy(), grp[C.Score].to_numpy())
    return out


def relevant_sets(ground_truth: pd.DataFrame) -> dict[object, set]:
    """From an interactions DataFrame build ``{user_id: {relevant item_id}}``."""
    out: dict[object, set] = {}
    for user, grp in ground_truth.groupby(C.User, sort=False):
        out[user] = set(grp[C.Item].tolist())
    return out
