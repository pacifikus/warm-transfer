"""
Thin benchmark entry-point built on the new abstractions.

All orchestration lives in BenchmarkRunner. This file only declares *what* to
run: which dataset, which model (+ its hyperparameters), and which methods.
Per-dataset BPR hyperparameters that genuinely differ (sparse Amazon needs more
iterations / higher LR / weaker reg than dense ML-1M) live in DATASET_CONFIGS.

Run from the repo root:
    python benchmarks/run.py                  # default: ml1m
    python benchmarks/run.py --dataset amazon
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
from warmtransfer.methods import (
    Random,
    MostPopular,
    GroupedMostPopular,
    ScoreAverage,
    ScoreCalibrated,
    EmbeddingAverage,
)

K = 10  # neighbours for the aggregation methods

# Per-dataset settings. Only things that genuinely differ live here; everything
# else is a BenchmarkRunner default.
DATASET_CONFIGS = {
    "ml1m": {
        "n_threshold": 50,
        "max_cold_eval": None,
        "results_path": "benchmarks/results_ml1m.csv",
        "bpr": dict(factors=64, iterations=30, learning_rate=0.01,
                    regularization=0.01, min_user_interactions=0),
    },
    "amazon": {
        "n_threshold": 50,
        "max_cold_eval": 8000,   # large cold set — cap for tractable eval
        "results_path": "benchmarks/results_amazon_toys.csv",
        # Tuned for sparse data: more iterations, higher LR, weaker reg.
        "bpr": dict(factors=64, iterations=200, learning_rate=0.05,
                    regularization=0.001, min_user_interactions=5),
    },
}


def build_methods() -> list:
    """The fixed method roster benchmarked on every dataset."""
    return [
        Random(),
        MostPopular(),
        GroupedMostPopular(),
        ScoreAverage(k=K),
        ScoreCalibrated(k=K, method="logistic"),
        EmbeddingAverage(k=K),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-Transfer benchmark")
    parser.add_argument(
        "--dataset", choices=list(DATASET_CONFIGS), default="ml1m",
        help="Dataset to benchmark (default: ml1m)",
    )
    args = parser.parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    bpr_kwargs = cfg["bpr"]

    runner = BenchmarkRunner(
        dataset=get_dataset(args.dataset),
        model_factory=lambda: BPRModel(**bpr_kwargs),
        methods=build_methods(),
        ks=[1, 5, 10],
        n_threshold=cfg["n_threshold"],
        max_cold_eval=cfg["max_cold_eval"],
        relaimpr_baseline="Random",
        results_path=cfg["results_path"],
        seed=42,
    )
    runner.run()


if __name__ == "__main__":
    main()
