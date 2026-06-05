"""
Baseline cold-start methods, exposed through the ColdStartMethod interface.

  Random              — uniform random scores (RelaImpr anchor / floor)
  MostPopular         — global engagement rate, same for every user (no personalisation)
  GroupedMostPopular  — category-conditioned user popularity (the strong baseline)
  ScoreAverage        — unweighted mean of k nearest warm neighbours' scores (no calibration)

The math inside each predict() is unchanged from the original baseline functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from warmtransfer.core.method import ColdStartMethod


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------

class Random(ColdStartMethod):
    """Uniform random score for every (user, item) pair."""

    name = "Random"

    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        self.seed = seed

    def predict(self, cold_df: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        out = cold_df.copy()
        out["predicted_score"] = rng.uniform(0, 1, len(out)).astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# Most Popular
# ---------------------------------------------------------------------------

class MostPopular(ColdStartMethod):
    """
    Global engagement rate for every cold item, perturbed slightly so per-item
    user ranking is non-trivial. No personalisation — a floor for personalised
    methods.
    """

    name = "Most Popular"

    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        self.seed = seed

    def predict(self, cold_df: pd.DataFrame) -> pd.DataFrame:
        global_mean = self.context.warm_df["engagement"].mean()
        rng = np.random.default_rng(self.seed)
        out = cold_df.copy()
        out["predicted_score"] = (
            global_mean + rng.normal(0, 1e-3, len(out))
        ).astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# Grouped Most Popular
# ---------------------------------------------------------------------------

class GroupedMostPopular(ColdStartMethod):
    """
    Category-conditioned popularity baseline.

    For each (user, cold_item) pair:
        score = mean engagement of this user with warm items that share at least
                one category with the cold item.
    Fallback chain: user-category rate -> global category rate -> global mean.

    Stronger than plain Most Popular because it accounts for item type: a user
    who liked Building Sets scores higher on a NEW Building Set than a new Card
    Game. If our method cannot beat this, we are merely recovering "user has
    history in this category" rather than finer item similarity.
    """

    name = "Grouped Most Popular"

    def fit(self, context) -> "GroupedMostPopular":
        super().fit(context)
        item_features = context.item_features
        warm_df = context.warm_df

        # Detect categorical columns: cat_* (Amazon) or genre_* (ML-1M)
        cat_cols = [
            c for c in item_features.columns
            if c.startswith("cat_") or c.startswith("genre_")
        ]
        self._cat_cols = cat_cols
        self._global_mean = float(warm_df["engagement"].mean())

        if not cat_cols:
            print("    No category/genre columns — will fall back to Most Popular")
            self._item_to_cats = {}
            self._user_cat_rate = {}
            self._cat_rate = {}
            return self
        print(f"    Using {len(cat_cols)} category columns for grouping")

        feat_mat = item_features[cat_cols].values
        item_ids_arr = item_features["item_id"].tolist()
        self._item_to_cats = {
            iid: np.flatnonzero(feat_mat[i] > 0).tolist()
            for i, iid in enumerate(item_ids_arr)
        }

        warm_with_cats = warm_df[["user_id", "item_id", "engagement"]].copy()
        warm_with_cats["cats"] = warm_with_cats["item_id"].map(self._item_to_cats)
        warm_with_cats = warm_with_cats[warm_with_cats["cats"].str.len() > 0]
        warm_exp = warm_with_cats.explode("cats")
        warm_exp["cats"] = warm_exp["cats"].astype(int)

        self._user_cat_rate = (
            warm_exp.groupby(["user_id", "cats"])["engagement"].mean().to_dict()
        )
        self._cat_rate = warm_exp.groupby("cats")["engagement"].mean().to_dict()
        return self

    def predict(self, cold_df: pd.DataFrame) -> pd.DataFrame:
        out = cold_df.copy()

        # No categorical features -> degenerate to plain Most Popular
        if not self._cat_cols:
            rng = np.random.default_rng(42)
            out["predicted_score"] = (
                self._global_mean + rng.normal(0, 1e-3, len(out))
            ).astype(np.float32)
            return out

        scores = np.zeros(len(cold_df), dtype=np.float32)
        user_ids = cold_df["user_id"].values
        item_ids = cold_df["item_id"].values

        for i in tqdm(range(len(cold_df)), desc="Grouped most popular"):
            uid = user_ids[i]
            iid = item_ids[i]
            cats = self._item_to_cats.get(iid, [])
            if not cats:
                scores[i] = self._global_mean
                continue
            vals = []
            for c in cats:
                r = self._user_cat_rate.get((uid, c))
                if r is None:
                    r = self._cat_rate.get(c, self._global_mean)
                vals.append(r)
            scores[i] = float(np.mean(vals))

        out["predicted_score"] = scores
        return out


# ---------------------------------------------------------------------------
# Score Average (direct, no calibration)
# ---------------------------------------------------------------------------

class ScoreAverage(ColdStartMethod):
    """
    Unweighted mean of the k nearest warm neighbours' model scores. Shows the
    raw gain from neighbour aggregation before any calibration is applied.
    """

    def __init__(self, k: int = 10) -> None:
        super().__init__()
        self.k = k
        self.name = f"Score Avg (k={k})"

    def predict(self, cold_df: pd.DataFrame) -> pd.DataFrame:
        ctx = self.context
        model = ctx.model
        similarity = ctx.similarity

        records: list[dict] = []
        cand_idx = similarity.precompute_candidate_matrix(ctx.warm_item_ids)
        grouped = cold_df.groupby("item_id", sort=False)

        for cold_item_id, item_df in tqdm(grouped, desc="Score average"):
            user_ids = item_df["user_id"].tolist()
            engagements = item_df["engagement"].tolist()

            neighbors = similarity.get_k_nearest(
                cold_item_id, k=self.k, candidate_index=cand_idx
            )
            if not neighbors:
                scores = np.full(len(user_ids), 0.5, dtype=np.float32)
            else:
                neighbor_ids = [n[0] for n in neighbors]
                all_u, all_i = [], []
                for uid in user_ids:
                    all_u.extend([uid] * len(neighbor_ids))
                    all_i.extend(neighbor_ids)
                raw = model.predict_batch(all_u, all_i).reshape(
                    len(user_ids), len(neighbor_ids)
                )
                scores = raw.mean(axis=1).astype(np.float32)

            for uid, score, eng in zip(user_ids, scores.tolist(), engagements):
                records.append({
                    "user_id": uid,
                    "item_id": cold_item_id,
                    "engagement": eng,
                    "predicted_score": score,
                })
        return pd.DataFrame(records)
