"""
Content-based item similarity index.

Builds a cosine-similarity lookup over item feature vectors so that
the calibrator can find the k warm neighbours of any cold item.

Typical usage
-------------
    sim = ItemSimilarity(item_features_df)

    # Find 10 warm items most similar to a cold item
    neighbours = sim.get_k_nearest(
        item_id=cold_item_id,
        candidate_ids=warm_item_ids,
        k=10,
    )
    # → [(item_id, similarity_score), ...]  sorted descending
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


@dataclass
class _CandidateIndex:
    """
    Pre-built lookup structure for a fixed candidate pool.
    Created by ItemSimilarity.precompute_candidate_matrix().
    """
    ids: list          # candidate item IDs (in matrix row order)
    matrix: np.ndarray  # L2-normalised feature matrix, shape (n_candidates, dim)


class ItemSimilarity:
    """
    Cosine-similarity lookup over L2-normalised item feature vectors.

    After normalisation, cosine similarity reduces to a simple dot product,
    so lookups are fast even for thousands of items.

    Parameters
    ----------
    item_features_df : DataFrame with an 'item_id' column plus numeric
                       feature columns (e.g. genre_Action, genre_Comedy, …).
    feature_cols     : which columns to use as features.
                       Default: all columns except 'item_id'.
    """

    def __init__(
        self,
        item_features_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> None:
        self.item_ids: list = item_features_df["item_id"].tolist()
        self.item_id_to_idx: dict = {iid: i for i, iid in enumerate(self.item_ids)}

        if feature_cols is None:
            feature_cols = [c for c in item_features_df.columns if c != "item_id"]
        self.feature_cols: list[str] = feature_cols

        X = item_features_df[feature_cols].values.astype(np.float32)

        # Handle zero-norm rows (items with no features) — set to uniform vector
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        zero_mask = (norms.squeeze() == 0)
        if zero_mask.any():
            X[zero_mask] = 1.0 / np.sqrt(X.shape[1])

        # L2-normalise: cosine_similarity(a, b) == a_norm @ b_norm.T
        self.feature_matrix: np.ndarray = normalize(X, norm="l2")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def precompute_candidate_matrix(self, candidate_ids: list) -> "_CandidateIndex":
        """
        Pre-build a candidate matrix for a fixed pool (e.g. warm_item_ids).

        Use this when you need to call get_k_nearest many times against the
        same candidate pool — it avoids rebuilding the index on every call.

            idx = sim.precompute_candidate_matrix(warm_ids)
            for cold_id in cold_ids:
                neighbours = sim.get_k_nearest(cold_id, candidate_index=idx, k=10)
        """
        valid = [iid for iid in candidate_ids if iid in self.item_id_to_idx]
        indices = np.array([self.item_id_to_idx[iid] for iid in valid], dtype=np.int32)
        matrix = self.feature_matrix[indices]          # (n_candidates, dim)
        return _CandidateIndex(valid, matrix)

    def get_k_nearest(
        self,
        item_id,
        candidate_ids: list | None = None,
        k: int = 10,
        candidate_index: "_CandidateIndex | None" = None,
    ) -> list[tuple]:
        """
        Find the k items most similar to item_id among candidate_ids.

        Parameters
        ----------
        item_id         : query item (must exist in item_features_df)
        candidate_ids   : restrict search to this pool (e.g. warm_item_ids).
                          Pass None to search the entire index.
                          The query item itself is always excluded.
        k               : number of neighbours to return
        candidate_index : pre-built _CandidateIndex from precompute_candidate_matrix().
                          When supplied, candidate_ids is ignored and the pre-built
                          matrix is used directly — much faster for repeated calls.

        Returns
        -------
        List of (item_id, similarity_score) tuples sorted by descending similarity.
        Returns fewer than k items if candidate_ids is small.
        """
        if item_id not in self.item_id_to_idx:
            return []

        q_idx = self.item_id_to_idx[item_id]
        query_vec = self.feature_matrix[q_idx]          # (dim,)

        if candidate_index is not None:
            # Fast path: pre-built matrix, single dot-product call
            sims = candidate_index.matrix @ query_vec   # (n_candidates,)
            ids  = candidate_index.ids
        elif candidate_ids is not None:
            candidates = [
                iid for iid in candidate_ids
                if iid != item_id and iid in self.item_id_to_idx
            ]
            if not candidates:
                return []
            c_indices = [self.item_id_to_idx[iid] for iid in candidates]
            cand_matrix = self.feature_matrix[c_indices]
            sims = cand_matrix @ query_vec
            ids  = candidates
        else:
            sims = self.feature_matrix @ query_vec
            ids  = self.item_ids

        # Exclude the query item itself
        sims_arr = np.asarray(sims, dtype=np.float32)
        pairs = [
            (iid, float(sims_arr[i]))
            for i, iid in enumerate(ids)
            if iid != item_id
        ]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:k]

    def similarity_between(self, item_a, item_b) -> float:
        """Return cosine similarity between two items."""
        if item_a not in self.item_id_to_idx or item_b not in self.item_id_to_idx:
            return 0.0
        va = self.feature_matrix[self.item_id_to_idx[item_a]]
        vb = self.feature_matrix[self.item_id_to_idx[item_b]]
        return float(va @ vb)
