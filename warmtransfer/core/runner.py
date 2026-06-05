"""
BenchmarkRunner — fixed orchestration; methods/models/datasets are data.

This is the old run_benchmark.main() reorganised so that *what* you benchmark
(which dataset, which model, which methods) is passed in as arguments, while
*how* the benchmark runs (split -> validate -> train -> build context -> run
each method -> score) stays fixed here.

Typical use (from a thin entry-point)
-------------------------------------
    runner = BenchmarkRunner(
        dataset=get_dataset("ml1m"),
        model_factory=lambda: BPRModel(factors=64, iterations=200, ...),
        methods=[Random(), MostPopular(), GroupedMostPopular(),
                 ScoreAverage(k=10), ScoreCalibrated(k=10), EmbeddingAverage(k=10)],
        n_threshold=50,
    )
    results = runner.run()
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from warmtransfer.core.dataset import Dataset
from warmtransfer.core.method import ColdStartMethod, Context
from warmtransfer.core.model import Model
from warmtransfer.splitters.splitter import (
    split_warm_cold,
    sample_pseudo_cold_items,
    print_split_stats,
)
from warmtransfer.similarity.item_similarity import ItemSimilarity
from warmtransfer.metrics.metrics import (
    summarise_metrics,
    global_auc,
    print_results_table,
)

from typing import Callable

SEP = "=" * 70


class BenchmarkRunner:
    """
    Run a full cold-start benchmark for one (dataset, model, methods) triple.

    Parameters
    ----------
    dataset           : a Dataset instance (knows how to load its data)
    model_factory     : zero-arg callable returning a FRESH, unfitted Model.
                        A factory (not an instance) is required because the
                        generalisation check trains a separate model on 90 % of
                        the warm data before the final model is trained on 100 %.
    methods           : list of ColdStartMethod instances to benchmark
    ks                : cutoffs for ranking metrics (default [1, 5, 10])
    n_threshold       : min interactions for an item to count as warm
    min_cold_interactions : min interactions a cold item needs to be evaluable
    pseudo_cold_frac  : fraction of warm items used to train calibrator methods
    max_cold_eval     : cap on number of cold items evaluated (None = all)
    relaimpr_baseline : method name whose AUC anchors RelaImpr (default "Random")
    validate_model    : run the 90/10 generalisation check before training
    results_path      : optional CSV path to save the results table
    seed              : RNG seed
    """

    def __init__(
        self,
        dataset: Dataset,
        model_factory: Callable[[], Model],
        methods: list[ColdStartMethod],
        ks: list[int] | None = None,
        n_threshold: int = 50,
        min_cold_interactions: int = 5,
        pseudo_cold_frac: float = 0.15,
        max_cold_eval: int | None = None,
        relaimpr_baseline: str = "Random",
        validate_model: bool = True,
        results_path: str | None = None,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.model_factory = model_factory
        self.methods = methods
        self.ks = ks if ks is not None else [1, 5, 10]
        self.n_threshold = n_threshold
        self.min_cold_interactions = min_cold_interactions
        self.pseudo_cold_frac = pseudo_cold_frac
        self.max_cold_eval = max_cold_eval
        self.relaimpr_baseline = relaimpr_baseline
        self.validate_model = validate_model
        self.results_path = results_path
        self.seed = seed

    # ==================================================================
    # Public entry point
    # ==================================================================

    def run(self) -> dict[str, dict]:
        """Execute the full benchmark and return {method_name: metrics}."""
        print(SEP)
        print(f"  Warm-Transfer Benchmark — {self.dataset.name}")
        print(SEP)

        interactions, item_features = self._load()
        warm_df, cold_df, warm_ids, pseudo_cold_df = self._split(interactions)

        if self.validate_model:
            self._validate_generalisation(warm_df)

        model = self._train_model(warm_df)
        similarity = self._build_similarity(item_features, interactions)

        context = Context(
            warm_df=warm_df,
            pseudo_cold_df=pseudo_cold_df,
            warm_item_ids=warm_ids,
            item_features=item_features,
            model=model,
            similarity=similarity,
        )

        predictions = self._run_methods(context, cold_df)
        results = self._score(predictions)
        self._report(results)
        return results

    # ==================================================================
    # Pipeline steps
    # ==================================================================

    def _load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        print("\n[1] Loading data ...")
        interactions, item_features = self.dataset.load()
        print(f"  Interactions      : {len(interactions):,}")
        print(f"  Items w/ features : {len(item_features):,}")
        return interactions, item_features

    def _split(self, interactions: pd.DataFrame):
        print(f"\n[2] Splitting items (n_threshold={self.n_threshold}) ...")
        warm_df, cold_df, warm_ids, cold_ids = split_warm_cold(
            interactions,
            n_threshold=self.n_threshold,
            min_cold_interactions=self.min_cold_interactions,
            random_seed=self.seed,
        )
        print_split_stats(warm_df, cold_df, warm_ids, cold_ids)

        if len(cold_ids) == 0:
            raise RuntimeError("No cold items — lower n_threshold.")

        # Optionally cap cold items for faster evaluation on large datasets
        if self.max_cold_eval is not None and len(cold_ids) > self.max_cold_eval:
            rng = np.random.default_rng(self.seed)
            sampled = rng.choice(
                cold_ids, size=self.max_cold_eval, replace=False
            ).tolist()
            cold_df = cold_df[cold_df["item_id"].isin(sampled)]
            print(f"  Capped cold eval  : {self.max_cold_eval:,} "
                  f"(sampled from {len(cold_ids):,})")

        pseudo_cold_ids = sample_pseudo_cold_items(
            warm_ids, frac=self.pseudo_cold_frac, random_seed=self.seed
        )
        pseudo_cold_df = warm_df[warm_df["item_id"].isin(pseudo_cold_ids)]
        print(f"  Pseudo-cold train : {len(pseudo_cold_ids):,} items, "
              f"{len(pseudo_cold_df):,} interactions")

        return warm_df, cold_df, warm_ids, pseudo_cold_df

    def _validate_generalisation(self, warm_df: pd.DataFrame, val_frac: float = 0.1):
        """
        Honest check: train a fresh model on 90 % of warm rows, measure AUC on
        the held-out 10 %. Tells us whether the model genuinely learned user
        preferences (high val AUC) or just memorised the training matrix (~0.5).
        Model-agnostic: works for any Model via known_user_ids / known_item_ids.
        """
        print("\n[3] Validating model generalisation (90/10 hold-out) ...")
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(warm_df))
        val_size = int(len(warm_df) * val_frac)
        val_df = warm_df.iloc[perm[:val_size]]
        train_df = warm_df.iloc[perm[val_size:]]
        print(f"  Train rows : {len(train_df):,}  |  Val rows : {len(val_df):,}")

        model = self.model_factory()
        model.fit(train_df)

        users = val_df["user_id"].tolist()
        items = val_df["item_id"].tolist()
        labels = val_df["engagement"].values.astype(int)
        scores = model.predict_batch(users, items)

        # Evaluate only on pairs the model can actually score
        known_users = set(model.known_user_ids)
        known_items = set(model.known_item_ids)
        if known_users and known_items:
            mask = np.array([
                (u in known_users) and (i in known_items)
                for u, i in zip(users, items)
            ])
        else:
            mask = np.ones(len(users), dtype=bool)

        n_pos = int(labels[mask].sum())
        n_neg = int(mask.sum()) - n_pos
        if n_pos == 0 or n_neg == 0:
            print(f"  Cannot compute val AUC (pos={n_pos}, neg={n_neg})")
            return None

        val_auc = float(roc_auc_score(labels[mask], scores[mask]))
        print(f"  Scoreable pairs : {int(mask.sum()):,} "
              f"({n_pos:,} pos, {n_neg:,} neg)")
        print(f"  Val AUC         : {val_auc:.4f}  {self._verdict(val_auc)}")
        return val_auc

    @staticmethod
    def _verdict(val_auc: float) -> str:
        if val_auc < 0.55:
            return "(does not generalise — train AUC would be memorisation)"
        if val_auc < 0.65:
            return "(generalises weakly)"
        if val_auc < 0.75:
            return "(generalises moderately)"
        return "(generalises well)"

    def _train_model(self, warm_df: pd.DataFrame) -> Model:
        print("\n[4] Training final model on full warm set ...")
        model = self.model_factory()
        model.fit(warm_df)
        return model

    def _build_similarity(
        self, item_features: pd.DataFrame, interactions: pd.DataFrame
    ) -> ItemSimilarity:
        print("\n[5] Building item-similarity index ...")
        # Restrict to items that actually appear in interactions
        interacted = set(interactions["item_id"].unique())
        feat_df = item_features[item_features["item_id"].isin(interacted)]
        similarity = ItemSimilarity(feat_df)
        print(f"  Feature dims : {len(similarity.feature_cols)}")
        return similarity

    def _run_methods(
        self, context: Context, cold_df: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        print("\n[6] Running methods ...")
        predictions: dict[str, pd.DataFrame] = {}
        for method in self.methods:
            if method.requires_embeddings and not context.model.supports_embeddings:
                print(f"  - {method.name}: SKIPPED "
                      f"(model has no embeddings)")
                continue
            print(f"  - {method.name} ...")
            method.fit(context)
            predictions[method.name] = method.predict(cold_df)
        return predictions

    def _score(self, predictions: dict[str, pd.DataFrame]) -> dict[str, dict]:
        print("\n[7] Computing metrics ...")
        # RelaImpr anchor: AUC of the named baseline method (default Random)
        base_preds = predictions.get(self.relaimpr_baseline)
        baseline_auc = global_auc(base_preds) if base_preds is not None else 0.5

        results: dict[str, dict] = {}
        for name, preds in predictions.items():
            results[name] = summarise_metrics(
                preds, ks=self.ks, baseline_auc=baseline_auc
            )
        return results

    def _report(self, results: dict[str, dict]) -> None:
        print("\n" + SEP)
        print(f"  RESULTS — {self.dataset.name}")
        print(SEP)
        print_results_table(results, ks=self.ks)

        if self.results_path:
            os.makedirs(os.path.dirname(self.results_path) or ".", exist_ok=True)
            pd.DataFrame(results).T.to_csv(self.results_path)
            print(f"\nResults saved -> {self.results_path}")
        print(SEP)
