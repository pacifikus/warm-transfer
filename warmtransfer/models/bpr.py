"""
BPRModel — Bayesian Personalised Ranking wrapper, implementing the Model ABC.

Uses the `implicit` library (the maintained replacement for LightFM on
Python 3.11+ / Windows). Same BPR objective, same user/item embedding layout.

This wrapper satisfies the full Model interface, including embedding access,
so it can drive both score-based and embedding-based cold-start methods.

    model = BPRModel(factors=64, iterations=200, min_user_interactions=5)
    model.fit(warm_df)                       # interface: fit(warm_df) only
    scores = model.predict_batch(users, items)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from implicit.bpr import BayesianPersonalizedRanking

from warmtransfer.core.model import Model


class BPRModel(Model):
    """
    Thin wrapper around implicit.bpr.BayesianPersonalizedRanking.

    Maps external user/item IDs <-> internal integer indices and exposes the
    Model interface. All training hyperparameters live in __init__ so that
    fit(warm_df) stays uniform across model families.

    Parameters
    ----------
    factors               : embedding dimension
    iterations            : training epochs
    learning_rate         : SGD step size
    regularization        : L2 weight decay
    min_user_interactions : drop users with fewer warm interactions than this
                            before training (BPR cannot learn from 1-2 examples
                            on sparse data). 0 = keep all users.
    """

    name = "BPR"

    def __init__(
        self,
        factors: int = 64,
        iterations: int = 30,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        min_user_interactions: int = 0,
    ) -> None:
        self.factors = factors
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.min_user_interactions = min_user_interactions

        self._model: BayesianPersonalizedRanking | None = None
        self._user_id_to_idx: dict = {}
        self._item_id_to_idx: dict = {}
        self._user_ids: list = []
        self._item_ids: list = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, warm_df: pd.DataFrame) -> "BPRModel":
        """Train on warm-item interactions [user_id, item_id, engagement]."""
        df = warm_df

        # Filter sparse users before building embeddings
        if self.min_user_interactions > 0:
            counts = df.groupby("user_id")["user_id"].transform("count")
            before = df["user_id"].nunique()
            df = df[counts >= self.min_user_interactions]
            after = df["user_id"].nunique()
            if before != after:
                print(
                    f"  Dropped {before - after:,} users with "
                    f"<{self.min_user_interactions} warm interactions "
                    f"({after:,} users remaining)"
                )

        self._user_ids = sorted(df["user_id"].unique().tolist())
        self._item_ids = sorted(df["item_id"].unique().tolist())
        self._user_id_to_idx = {uid: i for i, uid in enumerate(self._user_ids)}
        self._item_id_to_idx = {iid: i for i, iid in enumerate(self._item_ids)}

        n_users = len(self._user_ids)
        n_items = len(self._item_ids)

        rows = df["user_id"].map(self._user_id_to_idx).values
        cols = df["item_id"].map(self._item_id_to_idx).values
        # Only positive interactions: BPR's built-in negative sampling handles
        # unobserved items. Including explicit negatives confuses the ranking
        # objective when the negative rate is very low.
        pos_mask = df["engagement"].values > 0
        rows, cols = rows[pos_mask], cols[pos_mask]
        data = np.ones(len(rows), dtype=np.float32)

        # BPR in implicit expects user x item
        user_item = coo_matrix(
            (data, (rows, cols)), shape=(n_users, n_items)
        ).tocsr()

        print(
            f"  Training BPR (factors={self.factors}, "
            f"iterations={self.iterations})  |  "
            f"{n_users:,} users, {n_items:,} items, "
            f"{len(rows):,} positive interactions"
        )

        self._model = BayesianPersonalizedRanking(
            factors=self.factors,
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            verify_negative_samples=True,
        )
        self._model.fit(user_item)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_batch(self, user_ids: list, item_ids: list) -> np.ndarray:
        """
        Scores for (user_id, item_id) pairs = dot product of embeddings.
        Unknown IDs return 0.0.
        """
        assert self._model is not None, "Call fit() first"
        assert len(user_ids) == len(item_ids)

        user_factors = self._model.user_factors
        item_factors = self._model.item_factors

        scores = np.zeros(len(user_ids), dtype=np.float32)
        for k, (uid, iid) in enumerate(zip(user_ids, item_ids)):
            u_idx = self._user_id_to_idx.get(uid)
            i_idx = self._item_id_to_idx.get(iid)
            if u_idx is not None and i_idx is not None:
                scores[k] = float(user_factors[u_idx] @ item_factors[i_idx])
        return scores

    def predict_score(self, user_id, item_id) -> float:
        """Convenience wrapper for a single pair."""
        return float(self.predict_batch([user_id], [item_id])[0])

    # ------------------------------------------------------------------
    # Embedding access (Model interface)
    # ------------------------------------------------------------------

    @property
    def supports_embeddings(self) -> bool:
        return True

    def get_item_embedding(self, item_id) -> np.ndarray | None:
        assert self._model is not None, "Call fit() first"
        idx = self._item_id_to_idx.get(item_id)
        return None if idx is None else self._model.item_factors[idx].copy()

    def get_user_embedding(self, user_id) -> np.ndarray | None:
        assert self._model is not None, "Call fit() first"
        idx = self._user_id_to_idx.get(user_id)
        return None if idx is None else self._model.user_factors[idx].copy()

    def get_item_embeddings_batch(self, item_ids: list) -> np.ndarray:
        """
        Embedding matrix for item_ids, shape (n, dim). Unknown items -> zeros.
        Note: implicit stores factors + 1 (last column is the bias term).
        """
        assert self._model is not None, "Call fit() first"
        dim = self._model.item_factors.shape[1]
        out = np.zeros((len(item_ids), dim), dtype=np.float32)
        for k, iid in enumerate(item_ids):
            idx = self._item_id_to_idx.get(iid)
            if idx is not None:
                out[k] = self._model.item_factors[idx]
        return out

    def get_user_embeddings_batch(self, user_ids: list) -> np.ndarray:
        """
        Embedding matrix for user_ids, shape (n, dim). Unknown users -> zeros.
        Note: implicit stores factors + 1 (last column is the bias term).
        """
        assert self._model is not None, "Call fit() first"
        dim = self._model.user_factors.shape[1]
        out = np.zeros((len(user_ids), dim), dtype=np.float32)
        for k, uid in enumerate(user_ids):
            idx = self._user_id_to_idx.get(uid)
            if idx is not None:
                out[k] = self._model.user_factors[idx]
        return out

    # ------------------------------------------------------------------
    # Known-ID introspection (Model interface)
    # ------------------------------------------------------------------

    @property
    def known_user_ids(self) -> list:
        return self._user_ids

    @property
    def known_item_ids(self) -> list:
        return self._item_ids
