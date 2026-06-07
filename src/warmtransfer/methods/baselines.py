"""Cold-start baselines: Random, MostPopular, GroupedMostPopular.

GroupedMostPopular is a strong baseline (popularity within the cold item's category)
that naive methods usually do NOT beat. The project's goal is to surpass exactly this one.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from scipy import sparse

from warmtransfer._pdutils import map_codes, unique_sorted
from warmtransfer.columns import Columns as C
from warmtransfer.methods.base import ColdStartMethod, cross_join_frame, register_method
from warmtransfer.seeding import make_rng
from warmtransfer.types import TransferInputs


def _popularity(train: pd.DataFrame) -> dict[object, float]:
    """Item popularity in train (number of interactions)."""
    counts = cast("pd.Series", train.groupby(C.Item).size())
    return {k: float(v) for k, v in counts.items()}


@register_method("random")
class RandomScorer(ColdStartMethod):
    """Random score for each (user, cold_item) pair. Expected AUC ≈ 0.5."""

    requires = frozenset()

    def __init__(self) -> None:
        super().__init__()
        self._seed = 0

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        self._seed = seed

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        rng = make_rng(self._seed)
        scores = rng.random((len(user_ids), len(cold_item_ids)))
        return cross_join_frame(np.asarray(user_ids), np.asarray(cold_item_ids), scores)


@register_method("most_popular")
class MostPopular(ColdStartMethod):
    """Item score = its popularity in train (the same for all users).

    For truly cold items the train popularity is zero → the method is uninformative
    (AUC ≈ 0.5). This is a fair floor baseline.
    """

    requires = frozenset({"train_interactions"})

    def __init__(self) -> None:
        super().__init__()
        self._pop: dict[object, float] = {}

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        self._pop = _popularity(inputs.train_interactions)

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        item_scores = np.array([float(self._pop.get(it, 0.0)) for it in cold_item_ids])
        scores = np.tile(item_scores, (len(user_ids), 1))
        return cross_join_frame(np.asarray(user_ids), np.asarray(cold_item_ids), scores)


@register_method("grouped_most_popular")
class GroupedMostPopular(ColdStartMethod):
    """Popularity within the cold item's category (genre).

    Genre popularity = sum of train popularities of warm items of that genre. The
    cold item score = sum of its genres' popularities. Leak-free: uses only train +
    static content. A strong baseline.
    """

    requires = frozenset({"train_interactions", "content"})

    def __init__(self) -> None:
        super().__init__()
        self._genre_pop: np.ndarray | None = None
        self._cold_scores: dict[object, float] = {}

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        warm = inputs.warm_features
        cold = inputs.cold_features
        if warm is None or cold is None:
            raise ValueError("GroupedMostPopular requires warm_features and cold_features")

        pop = _popularity(inputs.train_interactions)
        warm_pop = np.array([pop.get(it, 0.0) for it in warm.item_ids])
        warm_mat = np.asarray(warm.matrix, dtype=float)
        # genre g popularity = Σ_i pop[i] * [item i has genre g]
        genre_pop = warm_mat.T @ warm_pop  # [n_genres]
        self._genre_pop = genre_pop

        cold_mat = np.asarray(cold.matrix, dtype=float)
        cold_scores = cold_mat @ genre_pop  # [n_cold]
        self._cold_scores = {it: float(s) for it, s in zip(cold.item_ids, cold_scores, strict=True)}

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        item_scores = np.array([float(self._cold_scores.get(it, 0.0)) for it in cold_item_ids])
        scores = np.tile(item_scores, (len(user_ids), 1))
        return cross_join_frame(np.asarray(user_ids), np.asarray(cold_item_ids), scores)


@register_method("grouped_most_popular_pers")
class GroupedMostPopularPersonalized(ColdStartMethod):
    """Personalized GroupedMP: user affinity to the cold item's genres.

    score(u, i) = Σ_g [item i has genre g] · (how many times u interacted with warm
    items of genre g in train). A strong personalized baseline — the project's main
    target. Leak-free: only train + static content.
    """

    requires = frozenset({"train_interactions", "content"})

    def __init__(self) -> None:
        super().__init__()
        self._affinity: np.ndarray | None = None
        self._u_pos: dict = {}
        self._cold_mat: np.ndarray | None = None
        self._cold_pos: dict = {}

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        warm = inputs.warm_features
        cold = inputs.cold_features
        if warm is None or cold is None:
            raise ValueError("grouped_most_popular_pers requires warm_features and cold_features")

        warm_mat = np.asarray(warm.matrix, dtype=float)  # [n_warm, n_genres]
        w_pos = {it: j for j, it in enumerate(warm.item_ids)}

        train = inputs.train_interactions
        users = unique_sorted(train[C.User])
        self._u_pos = {u: i for i, u in enumerate(users)}

        # train contains only warm items → every element is present in w_pos
        rows = map_codes(train[C.User], self._u_pos)
        cols = map_codes(train[C.Item], w_pos)
        # sparse user×warm-item counts → user×genre affinity (avoids a dense [n_users,
        # n_warm] matrix that OOMs on large datasets like MIND).
        counts = sparse.coo_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(len(users), warm_mat.shape[0]),
        ).tocsr()
        self._affinity = np.asarray(counts @ warm_mat)  # [n_users, n_genres]

        self._cold_mat = np.asarray(cold.matrix, dtype=float)
        self._cold_pos = {it: r for r, it in enumerate(cold.item_ids)}

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        assert self._affinity is not None and self._cold_mat is not None
        user_ids = np.asarray(user_ids)
        cold_item_ids = np.asarray(cold_item_ids)

        u_rows = np.array([self._u_pos.get(u, -1) for u in user_ids])
        known = u_rows >= 0
        affinity = np.zeros((len(user_ids), self._affinity.shape[1]))
        affinity[known] = self._affinity[u_rows[known]]

        c_rows = [self._cold_pos[it] for it in cold_item_ids]
        cold_mat = self._cold_mat[c_rows]  # [n_cold, n_genres]
        scores = affinity @ cold_mat.T  # [n_users, n_cold]
        return cross_join_frame(user_ids, cold_item_ids, scores)
