"""Тесты технических метрик раннера: latency и RSS."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from coldbench.adapters.base import ModelAdapter, register_adapter
from coldbench.config import BenchConfig, ComponentCfg, SplitterCfg
from coldbench.datasets.base import DatasetLoader, register_dataset
from coldbench.runner import BenchmarkRunner
from coldbench.splitters.base import Splitter, register_splitter
from coldscore.columns import Columns as C
from coldscore.methods.base import ColdStartMethod, cross_join_frame, register_method
from coldscore.types import Dataset, ItemFeatures, SplitResult, TransferInputs


def _features() -> ItemFeatures:
    return ItemFeatures(
        item_ids=np.array([10, 11, 20]),
        matrix=np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        feature_names=["a", "b"],
    )


@register_dataset("resource_test_dataset")
class ResourceTestDataset(DatasetLoader):
    def load(self) -> Dataset:
        interactions = pd.DataFrame(
            {
                C.User: [1, 1, 2],
                C.Item: [10, 20, 20],
                C.Weight: [1.0, 1.0, 1.0],
                C.Datetime: [0, 0, 0],
            }
        )
        return Dataset(interactions=interactions, item_features=_features(), name=self.name)


@register_splitter("resource_test_splitter")
class ResourceTestSplitter(Splitter):
    def split(self, dataset: Dataset, seed: int = 0) -> SplitResult:
        del seed
        train = cast(
            "pd.DataFrame",
            dataset.interactions[dataset.interactions[C.Item].isin([10, 11])],
        ).copy()
        test = cast(
            "pd.DataFrame",
            dataset.interactions[dataset.interactions[C.Item].isin([20])],
        ).copy()
        return SplitResult(
            train=train,
            val=pd.DataFrame(columns=dataset.interactions.columns),
            test=test,
            warm_items=np.array([10, 11]),
            cold_items=np.array([20]),
        )


@register_adapter("resource_test_adapter")
class ResourceTestAdapter(ModelAdapter):
    def fit(self, dataset: Dataset, seed: int = 0) -> ModelAdapter:
        del dataset, seed
        return self

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> pd.DataFrame:
        scores = np.ones((len(user_ids), len(item_ids)), dtype=float)
        return cross_join_frame(user_ids, item_ids, scores)


@register_method("resource_test_method")
class ResourceTestMethod(ColdStartMethod):
    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        del inputs, seed

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        scores = np.ones((len(user_ids), len(cold_item_ids)), dtype=float)
        return cross_join_frame(user_ids, cold_item_ids, scores)


def test_runner_records_resource_metrics() -> None:
    cfg = BenchConfig(
        datasets=["resource_test_dataset"],
        donors=[ComponentCfg(name="resource_test_adapter")],
        methods=[ComponentCfg(name="resource_test_method")],
        splitter=SplitterCfg(name="resource_test_splitter"),
        metrics_ks=(1,),
        max_eval_users=None,
    )

    records = BenchmarkRunner(cfg).run()

    assert len(records) == 1
    record = records[0]
    resource_cols = {
        "donor_fit_seconds",
        "donor_score_seconds",
        "method_fit_seconds",
        "method_predict_seconds",
        "rss_mb",
    }
    assert resource_cols <= set(record)
    for col in resource_cols:
        assert record[col] >= 0.0
