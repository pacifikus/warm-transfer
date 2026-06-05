"""
Feature-group ablation for the stacking calibrator.

Question this answers: how much of the calibrator's performance comes from the
model-transfer signal (group A) versus the model-independent popularity /
propensity priors (groups C, D)? The headline AUC alone cannot tell us — groups
C and D encode the same signal Grouped-Most-Popular exploits, so a strong full
result could be "laundered popularity" rather than real score transfer.

We run several calibrators that differ ONLY in which pre-registered feature
groups are active, all on the SAME trained model / similarity / split (one
BenchmarkRunner). Reading the AUC column down the variants isolates each group's
marginal contribution. This is analysis, not tuning: every variant is reported,
none is selected for being best.

Run from the repo root:
    python benchmarks/ablation.py                 # default: ml1m
    python benchmarks/ablation.py --dataset amazon
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datasets  # noqa: F401  (registers ml1m / amazon)
from warmtransfer.core.dataset import get_dataset
from warmtransfer.core.runner import BenchmarkRunner
from warmtransfer.models.bpr import BPRModel
from warmtransfer.methods import GroupedMostPopular, ScoreAverage, ScoreCalibrated

K = 10

# Reuse the same per-dataset BPR settings as the main benchmark.
DATASET_CONFIGS = {
    "ml1m": {
        "n_threshold": 50,
        "max_cold_eval": None,
        "results_path": "benchmarks/ablation_ml1m.csv",
        "bpr": dict(factors=64, iterations=30, learning_rate=0.01,
                    regularization=0.01, min_user_interactions=0),
    },
    "amazon": {
        "n_threshold": 50,
        "max_cold_eval": 8000,
        "results_path": "benchmarks/ablation_amazon_toys.csv",
        "bpr": dict(factors=64, iterations=200, learning_rate=0.05,
                    regularization=0.001, min_user_interactions=5),
    },
}


def build_methods() -> list:
    """Calibrator variants differing only by active feature groups, + references."""
    return [
        # references for context
        ScoreAverage(k=K),
        GroupedMostPopular(),
        # ablation variants (same model, same split)
        ScoreCalibrated(k=K, groups=("A",),                name="Score[A]  transfer only"),
        ScoreCalibrated(k=K, groups=("A", "B"),            name="Score[A+B]  transfer+trust"),
        ScoreCalibrated(k=K, groups=("C", "D"),            name="Score[C+D]  priors only (no model)"),
        ScoreCalibrated(k=K, groups=("B", "C", "D"),       name="Score[B+C+D]  no transfer"),
        ScoreCalibrated(k=K, groups=("A", "B", "C", "D"),  name="Score[full]"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrator feature-group ablation")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS), default="ml1m")
    args = parser.parse_args()
    cfg = DATASET_CONFIGS[args.dataset]

    runner = BenchmarkRunner(
        dataset=get_dataset(args.dataset),
        model_factory=lambda: BPRModel(**cfg["bpr"]),
        methods=build_methods(),
        ks=[1, 5, 10],
        n_threshold=cfg["n_threshold"],
        max_cold_eval=cfg["max_cold_eval"],
        relaimpr_baseline="Score Avg (k=10)",
        results_path=cfg["results_path"],
        seed=42,
    )
    runner.run()


if __name__ == "__main__":
    main()
