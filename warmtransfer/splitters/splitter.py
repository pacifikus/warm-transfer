"""
Item-based train / evaluation split for cold-start recommendation.

Core idea
---------
Items are divided by how many interactions they have accumulated:

  warm items  (>= n_threshold interactions)
      → the recommendation model is trained on these interactions only

  cold items  (< n_threshold, but >= min_cold_interactions)
      → interactions are withheld during model training and used to
        evaluate cold-start predictions

Additionally, a small fraction of warm items can be designated as
"pseudo-cold" for calibrator training: the base model has already seen
them, but we simulate the cold-start scenario by asking it to predict
their scores using only neighbour information — the same way it would
behave for a truly cold item.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Primary split: warm vs cold items
# ---------------------------------------------------------------------------

def split_warm_cold(
    interactions_df: pd.DataFrame,
    n_threshold: int = 50,
    min_cold_interactions: int = 5,
    random_seed: int = 42,  # kept for API consistency
) -> tuple[pd.DataFrame, pd.DataFrame, list, list]:
    """
    Split items into warm (old) and cold (new) sets based on interaction count.

    Parameters
    ----------
    interactions_df       : DataFrame with columns [user_id, item_id, timestamp, engagement]
    n_threshold           : items with >= this many interactions are warm
    min_cold_interactions : cold items must have at least this many interactions
                            (need enough ground-truth to evaluate)
    random_seed           : unused, kept for API consistency

    Returns
    -------
    warm_df        : interactions of warm items  → train the rec model on this
    cold_df        : interactions of cold items  → evaluate cold-start predictions
    warm_item_ids  : list of warm item IDs
    cold_item_ids  : list of cold item IDs
    """
    item_counts = interactions_df.groupby("item_id").size()

    warm_item_ids: list = item_counts[item_counts >= n_threshold].index.tolist()
    cold_item_ids: list = item_counts[
        (item_counts < n_threshold) & (item_counts >= min_cold_interactions)
    ].index.tolist()

    warm_df = (
        interactions_df[interactions_df["item_id"].isin(warm_item_ids)]
        .reset_index(drop=True)
    )
    cold_df = (
        interactions_df[interactions_df["item_id"].isin(cold_item_ids)]
        .reset_index(drop=True)
    )

    return warm_df, cold_df, warm_item_ids, cold_item_ids


# ---------------------------------------------------------------------------
# Pseudo-cold sampling: for calibrator training
# ---------------------------------------------------------------------------

def sample_pseudo_cold_items(
    warm_item_ids: list,
    frac: float = 0.15,
    random_seed: int = 42,
) -> list:
    """
    Sample a subset of warm items to serve as pseudo-cold items when
    training the calibrator.

    The base model has seen these items during training, so we have their
    ground-truth interactions. We pretend each is cold (ignoring its direct
    embedding) and ask the calibrator to predict via neighbour aggregation.
    The actual interactions provide the supervision signal.

    Parameters
    ----------
    warm_item_ids : all warm item IDs
    frac          : fraction to hold out as pseudo-cold (default 15 %)
    random_seed   : for reproducibility

    Returns
    -------
    pseudo_cold_ids : list of item IDs to treat as pseudo-cold
    """
    rng = np.random.default_rng(random_seed)
    n = max(10, int(len(warm_item_ids) * frac))
    n = min(n, len(warm_item_ids))
    return rng.choice(warm_item_ids, size=n, replace=False).tolist()


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------

def print_split_stats(
    warm_df: pd.DataFrame,
    cold_df: pd.DataFrame,
    warm_item_ids: list,
    cold_item_ids: list,
) -> None:
    """Print a concise summary of the warm / cold split."""
    print(f"  Warm items : {len(warm_item_ids):>5,}  |  {len(warm_df):>8,} interactions")
    print(f"  Cold items : {len(cold_item_ids):>5,}  |  {len(cold_df):>8,} interactions")

    if len(cold_df) > 0:
        cc = cold_df.groupby("item_id").size()
        print(f"  Cold item interaction counts — "
              f"min {cc.min()}, median {cc.median():.0f}, max {cc.max()}")
