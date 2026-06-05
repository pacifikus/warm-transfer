"""
Embedding-level cold-start: estimate a cold item's embedding as the
similarity-weighted mean of its k nearest warm neighbours' embeddings, then
score in the model's latent space.

    cold_emb = Σ sim(cold, n_j) * item_emb_n_j  /  Σ sim(cold, n_j)
    score(u, cold) = user_emb_u · cold_emb

Why this beats score aggregation in principle: it scores from inside the latent
space the model learned (place the cold item at its neighbours' centroid, then
score), rather than averaging already-collapsed scalar scores.

This module keeps the original EmbeddingAggregator engine verbatim and wraps it
in the ColdStartMethod interface as `EmbeddingAverage`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from warmtransfer.core.method import ColdStartMethod


# ===========================================================================
# Engine (unchanged math)
# ===========================================================================

class EmbeddingAggregator:
    """Cold-start scorer via similarity-weighted mean embedding aggregation."""

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def _estimate_cold_embedding(
        self, cold_item_id, warm_item_ids, model, item_similarity,
        candidate_index=None,
    ) -> np.ndarray | None:
        if candidate_index is not None:
            neighbors = item_similarity.get_k_nearest(
                cold_item_id, k=self.k, candidate_index=candidate_index
            )
        else:
            neighbors = item_similarity.get_k_nearest(
                cold_item_id, warm_item_ids, k=self.k
            )
        if not neighbors:
            return None

        neighbor_ids = [n[0] for n in neighbors]
        neighbor_sims = np.array([n[1] for n in neighbors], dtype=np.float32)

        neighbor_embs = model.get_item_embeddings_batch(neighbor_ids)

        sim_sum = neighbor_sims.sum()
        if sim_sum < 1e-9:
            cold_emb = neighbor_embs.mean(axis=0)
        else:
            cold_emb = (neighbor_embs * neighbor_sims[:, None]).sum(axis=0) / sim_sum
        return cold_emb.astype(np.float32)

    def predict_scores(
        self, cold_item_id, user_ids, warm_item_ids, model, item_similarity,
        candidate_index=None,
    ) -> np.ndarray:
        cold_emb = self._estimate_cold_embedding(
            cold_item_id, warm_item_ids, model, item_similarity,
            candidate_index=candidate_index,
        )
        if cold_emb is None:
            return np.zeros(len(user_ids), dtype=np.float32)

        user_embs = model.get_user_embeddings_batch(user_ids)
        scores = user_embs @ cold_emb
        return scores.astype(np.float32)

    def predict_all_cold(
        self, cold_df, warm_item_ids, model, item_similarity,
    ) -> pd.DataFrame:
        records: list[dict] = []
        cand_idx = item_similarity.precompute_candidate_matrix(warm_item_ids)
        grouped = cold_df.groupby("item_id", sort=False)

        for cold_item_id, item_df in tqdm(grouped, desc="EmbeddingAggregator predict"):
            user_ids = item_df["user_id"].tolist()
            engagements = item_df["engagement"].tolist()
            scores = self.predict_scores(
                cold_item_id, user_ids, warm_item_ids, model, item_similarity,
                candidate_index=cand_idx,
            )
            for uid, score, eng in zip(user_ids, scores.tolist(), engagements):
                records.append({
                    "user_id": uid,
                    "item_id": cold_item_id,
                    "engagement": eng,
                    "predicted_score": score,
                })
        return pd.DataFrame(records)


# ===========================================================================
# ColdStartMethod wrapper
# ===========================================================================

class EmbeddingAverage(ColdStartMethod):
    """Embedding-aggregation cold-start, exposed through the uniform interface."""

    requires_embeddings = True

    def __init__(self, k: int = 10) -> None:
        super().__init__()
        self.k = k
        self.name = f"Embedding Avg (k={k})"
        self._engine = EmbeddingAggregator(k=k)

    def predict(self, cold_df: pd.DataFrame) -> pd.DataFrame:
        ctx = self.context
        return self._engine.predict_all_cold(
            cold_df,
            warm_item_ids=ctx.warm_item_ids,
            model=ctx.model,
            item_similarity=ctx.similarity,
        )
