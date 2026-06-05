"""
Score-level cold-start calibrator (stacking with meta-features).

For a cold item c and user u:
  1. Find the k warm items most content-similar to c.
  2. Ask the base model to score those k warm items for u (batch call).
  3. Build a heterogeneous feature vector (groups A/B/C/D, see RESEARCH.md §3.3).
  4. Run it through a trained logistic (or ridge) regression.

This is stacking with meta-features (Wolpert 1992; Sill et al. FWLS 2009;
Bao et al. STREAM, RecSys 2009) — a faithful reimplementation of a known
family, not a novel method. The feature set is PRE-REGISTERED (frozen before
looking at any result) and identical across datasets.

Training uses *pseudo-cold* items: warm items the model already learned, whose
real interactions provide labels while we simulate them as cold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from warmtransfer.core.method import ColdStartMethod


# ---------------------------------------------------------------------------
# Warm-world priors (model-independent, computed once on warm interactions)
# ---------------------------------------------------------------------------

class _WarmPriors:
    """
    User-side and item-side priors derived purely from warm interactions.

    These feed feature groups C and D. They are model-independent: a popularity
    / propensity signal that the meta-learner combines with the model-transfer
    signal (group A). Computed once on the warm world and reused at predict time
    so train and inference see the same prior values.
    """

    def __init__(self, warm_df: pd.DataFrame) -> None:
        by_user = warm_df.groupby("user_id")["engagement"]
        self.user_support = np.log1p(by_user.size()).to_dict()   # c_support
        self.user_rate = by_user.mean().to_dict()                # c_rate
        self.item_pop = warm_df.groupby("item_id")["engagement"].mean().to_dict()  # d_pop source

        # Defaults for users / items unseen in the warm world.
        self.default_support = 0.0
        self.default_rate = float(warm_df["engagement"].mean())
        self.default_pop = self.default_rate


# ---------------------------------------------------------------------------
# Feature builder — heterogeneous groups A / B / C / D
# ---------------------------------------------------------------------------

# Each feature belongs to exactly one pre-registered group (RESEARCH.md §3.3).
_GROUP_FEATURES = {
    "A": ["a_mean", "a_w"],            # model-transfer
    "B": ["b_meansim", "b_gap", "b_n"],  # neighbourhood quality
    "C": ["c_support", "c_rate"],      # user-side prior
    "D": ["d_pop"],                    # item-side popularity prior
}
ALL_GROUPS = ("A", "B", "C", "D")


def _feature_names(groups) -> list[str]:
    """Ordered feature names for the active groups (canonical A,B,C,D order)."""
    names: list[str] = []
    for g in ALL_GROUPS:
        if g in groups:
            names.extend(_GROUP_FEATURES[g])
    return names


def _build_feature_matrix(neighbors, user_ids, score_fn, priors: _WarmPriors, groups) -> np.ndarray:
    """
    Build a (n_users, n_active_features) feature matrix for one target item.

    Only the columns for the active feature `groups` (subset of A/B/C/D) are
    produced — this lets the ablation toggle whole groups on/off. Group A is the
    only one that queries the model, so it is skipped entirely when A is off.

    neighbors : list of (warm_item_id, similarity), sorted by descending sim.
    user_ids  : the users to score for this item.
    score_fn  : model.predict_batch(user_ids, item_ids) -> np.ndarray.
    priors    : warm-world user/item priors (groups C and D).
    groups    : iterable subset of {"A","B","C","D"}.
    """
    n_users = len(user_ids)
    width = len(_feature_names(groups))
    if not neighbors or n_users == 0:
        return np.zeros((n_users, width), dtype=np.float32)

    n_nbrs = len(neighbors)
    neighbor_ids = [n[0] for n in neighbors]
    neighbor_sims = np.array([n[1] for n in neighbors], dtype=np.float32)

    cols: dict[str, np.ndarray] = {}

    # --- Group A: model-transfer signal (only group that queries the model)-
    if "A" in groups:
        all_user_ids: list = []
        all_item_ids: list = []
        for uid in user_ids:
            all_user_ids.extend([uid] * n_nbrs)
            all_item_ids.extend(neighbor_ids)
        scores_matrix = score_fn(all_user_ids, all_item_ids).astype(np.float32).reshape(n_users, n_nbrs)
        sim_sum = float(neighbor_sims.sum())
        cols["a_mean"] = scores_matrix.mean(axis=1)
        cols["a_w"] = scores_matrix @ neighbor_sims / (sim_sum + 1e-8)

    # --- Group B: neighbourhood-quality meta-features (per item, broadcast)-
    if "B" in groups:
        cols["b_meansim"] = np.full(n_users, float(neighbor_sims.mean()), dtype=np.float32)
        cols["b_gap"] = np.full(n_users, float(neighbor_sims[0] - neighbor_sims[-1]), dtype=np.float32)
        cols["b_n"] = np.full(n_users, float(n_nbrs), dtype=np.float32)

    # --- Group C: user-side prior (per user) ------------------------------
    if "C" in groups:
        cols["c_support"] = np.array(
            [priors.user_support.get(u, priors.default_support) for u in user_ids],
            dtype=np.float32,
        )
        cols["c_rate"] = np.array(
            [priors.user_rate.get(u, priors.default_rate) for u in user_ids],
            dtype=np.float32,
        )

    # --- Group D: item-side popularity prior (per item, broadcast) --------
    if "D" in groups:
        d_pop = float(np.mean([priors.item_pop.get(i, priors.default_pop) for i in neighbor_ids]))
        cols["d_pop"] = np.full(n_users, d_pop, dtype=np.float32)

    features = np.column_stack([cols[name] for name in _feature_names(groups)])
    return features.astype(np.float32)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class WarmTransferCalibrator:
    """Stacking calibrator over heterogeneous meta-features (logistic or ridge)."""

    def __init__(self, k: int = 10, method: str = "logistic", groups=ALL_GROUPS) -> None:
        if method not in ("logistic", "ridge"):
            raise ValueError("method must be 'logistic' or 'ridge'")
        groups = tuple(g for g in ALL_GROUPS if g in set(groups))  # canonical order, validated
        if not groups:
            raise ValueError("at least one feature group (A/B/C/D) must be active")
        self.k = k
        self.method = method
        self.groups = groups
        self._model = None
        self._scaler = StandardScaler()
        self._priors: _WarmPriors | None = None
        self._is_fitted = False

    def fit(self, pseudo_cold_interactions, warm_item_ids, score_fn, item_similarity, warm_df):
        self._priors = _WarmPriors(warm_df)

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []

        cand_idx = item_similarity.precompute_candidate_matrix(warm_item_ids)

        items = pseudo_cold_interactions["item_id"].unique().tolist()
        for item_id in tqdm(items, desc="Building calibrator training data"):
            item_df = pseudo_cold_interactions[
                pseudo_cold_interactions["item_id"] == item_id
            ]
            user_ids = item_df["user_id"].tolist()
            labels = item_df["engagement"].values.astype(np.float32)

            neighbors = item_similarity.get_k_nearest(
                item_id, k=self.k, candidate_index=cand_idx
            )
            feats = _build_feature_matrix(neighbors, user_ids, score_fn, self._priors, self.groups)
            X_parts.append(feats)
            y_parts.append(labels)

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        X_scaled = self._scaler.fit_transform(X)

        if self.method == "logistic":
            self._model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        else:
            self._model = Ridge(alpha=1.0)

        self._model.fit(X_scaled, y)
        self._is_fitted = True

        n_pos = int(y.sum())
        print(
            f"Calibrator fitted  |  {len(X):,} training samples  |  "
            f"{n_pos:,} positive ({y.mean():.1%})"
        )
        return self

    def predict_all_cold(self, cold_df, warm_item_ids, score_fn, item_similarity):
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_all_cold()")

        records: list[dict] = []
        cand_idx = item_similarity.precompute_candidate_matrix(warm_item_ids)
        grouped = cold_df.groupby("item_id", sort=False)

        for cold_item_id, item_df in tqdm(grouped, desc="Predicting cold items"):
            user_ids = item_df["user_id"].tolist()
            engagements = item_df["engagement"].tolist()

            neighbors = item_similarity.get_k_nearest(
                cold_item_id, k=self.k, candidate_index=cand_idx
            )
            feats = _build_feature_matrix(neighbors, user_ids, score_fn, self._priors, self.groups)
            feats_scaled = self._scaler.transform(feats)
            if self.method == "logistic":
                scores = self._model.predict_proba(feats_scaled)[:, 1].astype(np.float32)
            else:
                scores = self._model.predict(feats_scaled).astype(np.float32)

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

class ScoreCalibrated(ColdStartMethod):
    """Stacking calibrator, exposed through the uniform interface."""

    def __init__(self, k: int = 10, method: str = "logistic", groups=ALL_GROUPS, name=None) -> None:
        super().__init__()
        self.k = k
        self.method = method
        self._engine = WarmTransferCalibrator(k=k, method=method, groups=groups)
        self.groups = self._engine.groups
        self.name = name or f"Score {method.capitalize()} (k={k})"

    def fit(self, context) -> "ScoreCalibrated":
        super().fit(context)
        self._engine.fit(
            context.pseudo_cold_df,
            warm_item_ids=context.warm_item_ids,
            score_fn=context.score_fn,
            item_similarity=context.similarity,
            warm_df=context.warm_df,
        )
        return self

    def predict(self, cold_df: pd.DataFrame) -> pd.DataFrame:
        ctx = self.context
        return self._engine.predict_all_cold(
            cold_df,
            warm_item_ids=ctx.warm_item_ids,
            score_fn=ctx.score_fn,
            item_similarity=ctx.similarity,
        )
