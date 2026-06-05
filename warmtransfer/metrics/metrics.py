"""
Evaluation metrics for the warm-transfer benchmark.

All top-level functions operate on a *predictions DataFrame* with columns:
    user_id, item_id, engagement (0/1), predicted_score (float)

Two evaluation perspectives are supported:

  Per-item  — for each cold item, rank users by predicted score and check
              whether the model identifies interacting users (AUC, NDCG, …).
              Mirrors the cold-start evaluation in the SIGIR 2021 MWUF paper.

  Global    — flatten all (user, item, engagement, score) rows into one pool
              and compute aggregate statistics.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Single-ranking helpers  (operate on 1-D numpy arrays)
# ---------------------------------------------------------------------------

def _dcg_at_k(y_true: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain for the top-k elements (already sorted)."""
    top_k = y_true[:k]
    positions = np.arange(2, len(top_k) + 2)           # positions 2, 3, …, k+1
    return float(np.sum(top_k / np.log2(positions)))


def _ideal_dcg_at_k(y_true: np.ndarray, k: int) -> float:
    """Ideal DCG: assumes all positives are ranked first."""
    ideal = np.sort(y_true)[::-1]
    return _dcg_at_k(ideal, k)


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """NDCG@k for one ranking."""
    if y_true.sum() == 0:
        return 0.0
    order = np.argsort(y_score)[::-1]
    idcg = _ideal_dcg_at_k(y_true, k)
    if idcg == 0:
        return 0.0
    return _dcg_at_k(y_true[order], k) / idcg


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Recall@k: fraction of positives captured in the top-k."""
    n_pos = y_true.sum()
    if n_pos == 0:
        return 0.0
    order = np.argsort(y_score)[::-1][:k]
    return float(y_true[order].sum() / n_pos)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Precision@k: fraction of top-k predictions that are positive."""
    order = np.argsort(y_score)[::-1][:k]
    return float(y_true[order].mean())


def map_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Mean Average Precision at k.
    Averages precision-at-rank over every position where a positive occurs
    (within top-k).
    """
    order = np.argsort(y_score)[::-1][:k]
    top_k_true = y_true[order]
    if top_k_true.sum() == 0:
        return 0.0
    n_relevant = y_true.sum()
    precisions = []
    running_tp = 0
    for rank, label in enumerate(top_k_true, start=1):
        if label == 1:
            running_tp += 1
            precisions.append(running_tp / rank)
    return float(sum(precisions) / min(k, n_relevant))


def mrr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean Reciprocal Rank: 1 / rank_of_first_positive."""
    order = np.argsort(y_score)[::-1]
    for rank, idx in enumerate(order, start=1):
        if y_true[idx] == 1:
            return 1.0 / rank
    return 0.0


def rela_impr(auc_measured: float, auc_baseline: float) -> float:
    """
    Relative improvement over a baseline AUC (from SIGIR 2021 MWUF paper).

        RelaImpr = (AUC_model − 0.5) / (AUC_baseline − 0.5) − 1  [× 100 %]

    Returns 0.0 when auc_baseline == 0.5 (random baseline).
    """
    if abs(auc_baseline - 0.5) < 1e-9:
        return 0.0
    return ((auc_measured - 0.5) / (auc_baseline - 0.5) - 1.0) * 100.0


# ---------------------------------------------------------------------------
# Per-item evaluation
# ---------------------------------------------------------------------------

def per_item_metrics(
    predictions_df: pd.DataFrame,
    ks: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute ranking metrics for each cold item independently.

    For each cold item we rank *users* by predicted score and measure how
    well the ranking recovers the users who actually interacted with the item.

    Returns
    -------
    DataFrame with one row per cold item:
        item_id, n_interactions, auc, recall@k, precision@k, map@k, ndcg@k, mrr
    """
    if ks is None:
        ks = [1, 5, 10]

    rows = []
    for item_id, group in predictions_df.groupby("item_id"):
        y_true = group["engagement"].values.astype(float)
        y_score = group["predicted_score"].values.astype(float)

        # Skip items where all labels are identical (AUC undefined)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue

        row: dict = {"item_id": item_id, "n_interactions": len(group)}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                row["auc"] = float(roc_auc_score(y_true, y_score))
            except Exception:
                row["auc"] = float("nan")

        for k in ks:
            row[f"recall@{k}"]    = recall_at_k(y_true, y_score, k)
            row[f"precision@{k}"] = precision_at_k(y_true, y_score, k)
            row[f"map@{k}"]       = map_at_k(y_true, y_score, k)
            row[f"ndcg@{k}"]      = ndcg_at_k(y_true, y_score, k)

        row["mrr"] = mrr(y_true, y_score)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Global / aggregate evaluation
# ---------------------------------------------------------------------------

def global_auc(predictions_df: pd.DataFrame) -> float:
    """AUC computed over the full pool of (user, item) pairs."""
    return float(roc_auc_score(
        predictions_df["engagement"],
        predictions_df["predicted_score"],
    ))


def summarise_metrics(
    predictions_df: pd.DataFrame,
    ks: list[int] | None = None,
    baseline_auc: float = 0.5,
) -> dict:
    """
    Compute all metrics and return a flat dict ready for a results table.

    Includes:
      - global AUC
      - per-item-averaged recall@k, precision@k, map@k, ndcg@k, mrr
      - RelaImpr relative to baseline_auc

    Parameters
    ----------
    predictions_df : DataFrame [user_id, item_id, engagement, predicted_score]
    ks             : list of k values (default [1, 5, 10])
    baseline_auc   : AUC of the reference method for RelaImpr (default 0.5 = random)
    """
    if ks is None:
        ks = [1, 5, 10]

    results: dict = {}

    # Global AUC
    try:
        results["auc"] = global_auc(predictions_df)
    except Exception:
        results["auc"] = float("nan")

    # Per-item averages
    per_item = per_item_metrics(predictions_df, ks=ks)
    if len(per_item) > 0:
        for col in per_item.columns:
            if col not in ("item_id", "n_interactions"):
                results[col] = float(per_item[col].mean())

    results["rela_impr"] = rela_impr(results.get("auc", 0.5), baseline_auc)
    return results


def print_results_table(
    method_results: dict[str, dict],
    ks: list[int] | None = None,
) -> None:
    """
    Pretty-print a results comparison table.

    Parameters
    ----------
    method_results : {method_name: metrics_dict}  (from summarise_metrics)
    ks             : k values to display
    """
    if ks is None:
        ks = [1, 5, 10]

    col_order = (
        ["auc"]
        + [f"ndcg@{k}" for k in ks]
        + [f"recall@{k}" for k in ks]
        + [f"map@{k}" for k in ks]
        + ["mrr", "rela_impr"]
    )

    df = pd.DataFrame(method_results).T
    available_cols = [c for c in col_order if c in df.columns]
    df = df[available_cols].astype(float).round(4)
    df.index.name = "method"

    print(df.to_string())
