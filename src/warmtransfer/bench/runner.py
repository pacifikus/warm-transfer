"""Раннер бенчмарка: датасеты × доноры × методы × сиды → таблица метрик.

Каждый прогон: честный сплит → обучение донора на warm → скоры донора → построение
TransferInputs → fit/predict каждого метода на (test_users × cold_items) → метрики.
"""

from __future__ import annotations

from time import perf_counter
from typing import cast

import numpy as np
import pandas as pd
import psutil

# импорт методов регистрирует их в реестре
from warmtransfer import methods as _methods_pkg  # noqa: F401
from warmtransfer._pdutils import unique_sorted
from warmtransfer.bench.adapters import adapters
from warmtransfer.bench.config import BenchConfig
from warmtransfer.bench.datasets import datasets
from warmtransfer.bench.splitters import splitters
from warmtransfer.columns import Columns as C
from warmtransfer.exceptions import MissingInputError
from warmtransfer.methods import methods
from warmtransfer.metrics import calc_metrics
from warmtransfer.seeding import make_rng
from warmtransfer.similarity import content_similarity
from warmtransfer.types import Dataset, SplitResult, TransferInputs


class BenchmarkRunner:
    """Прогоняет матрицу из ``BenchConfig`` и возвращает записи результатов."""

    def __init__(self, config: BenchConfig) -> None:
        self.config = config

    def run(self) -> list[dict]:
        records: list[dict] = []
        for dataset_name in self.config.datasets:
            ds = datasets.get(dataset_name)().load()
            for seed in self.config.seeds:
                records.extend(self._run_one(ds, dataset_name, seed))
        return records

    def _run_one(self, ds: Dataset, dataset_name: str, seed: int) -> list[dict]:
        cfg = self.config
        process = psutil.Process()
        splitter = splitters.get(cfg.splitter.name)(**cfg.splitter.params)
        split = splitter.split(ds, seed)

        if ds.item_features is None:
            raise ValueError(f"{dataset_name}: нужен item_features для cold-start методов")
        warm_features = ds.item_features.subset(split.warm_items)
        cold_features = ds.item_features.subset(split.cold_items)

        test_users = self._sample_test_users(split, seed)
        ground_truth = cast(
            "pd.DataFrame", split.test[split.test[C.User].isin(test_users.tolist())]
        )
        similarity = content_similarity(cold_features, warm_features)

        # val-cold фолд для супервизорных методов (если задан val_frac > 0)
        val_cold_features = None
        val_similarity = None
        val_interactions = None
        if len(split.val):
            val_items = unique_sorted(split.val[C.Item])
            val_cold_features = ds.item_features.subset(val_items)
            val_similarity = content_similarity(val_cold_features, warm_features)
            val_interactions = cast(
                "pd.DataFrame", split.val[split.val[C.User].isin(test_users.tolist())]
            )

        records: list[dict] = []
        for donor_cfg in cfg.donors:
            donor = adapters.get(donor_cfg.name)(**donor_cfg.params)
            t0 = perf_counter()
            donor.fit(Dataset(split.train, warm_features, name=dataset_name), seed=seed)
            donor_fit_seconds = perf_counter() - t0
            t0 = perf_counter()
            donor_scores = donor.score(test_users, split.warm_items)
            donor_score_seconds = perf_counter() - t0
            embeddings = donor.embeddings()  # None у доноров без латентного пространства

            inputs = TransferInputs(
                donor_scores=donor_scores,
                train_interactions=split.train,
                warm_features=warm_features,
                cold_features=cold_features,
                similarity=similarity,
                embeddings=embeddings,
                warm_items=split.warm_items,
                cold_items=split.cold_items,
                val_cold_features=val_cold_features,
                val_similarity=val_similarity,
                val_interactions=val_interactions,
            )

            for method_cfg in cfg.methods:
                method = methods.get(method_cfg.name)(**method_cfg.params)
                try:
                    t0 = perf_counter()
                    method.fit(inputs, seed=seed)
                    method_fit_seconds = perf_counter() - t0
                except MissingInputError as exc:
                    print(
                        f"[warmbench] пропуск {method_cfg.key} на доноре "
                        f"{donor_cfg.key}: {exc}"
                    )
                    continue
                t0 = perf_counter()
                reco = method.predict(test_users, split.cold_items)
                method_predict_seconds = perf_counter() - t0
                metric_vals = calc_metrics(reco, ground_truth, ks=cfg.metrics_ks)
                records.append(
                    {
                        "dataset": dataset_name,
                        "donor": donor_cfg.key,
                        "method": method_cfg.key,
                        "seed": seed,
                        "donor_fit_seconds": donor_fit_seconds,
                        "donor_score_seconds": donor_score_seconds,
                        "method_fit_seconds": method_fit_seconds,
                        "method_predict_seconds": method_predict_seconds,
                        "rss_mb": process.memory_info().rss / (1024 * 1024),
                        **metric_vals,
                    }
                )
        return records

    def _sample_test_users(self, split: SplitResult, seed: int) -> np.ndarray:
        users = unique_sorted(split.test[C.User])
        cap = self.config.max_eval_users
        if cap is not None and len(users) > cap:
            rng = make_rng(seed)
            users = np.sort(rng.choice(users, size=cap, replace=False))
        return users
