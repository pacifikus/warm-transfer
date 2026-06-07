"""Анализ recall@10 по бакетам популярности cold-айтемов (голова/хвост).

Прогоняет один cell (датасет × ALS), считает recall@10 по бакетам популярности cold-айтемов
для цели и сильных трансфер-методов. Показывает, где трансфер выигрывает — на массовых или
нишевых cold-айтемах.

Запуск: ``uv run python scripts/popularity_buckets.py [dataset]`` (по умолчанию ml-1m).
"""

from __future__ import annotations

import sys
from typing import cast

import numpy as np
import pandas as pd

from warmtransfer.bench.adapters.als import ALSAdapter
from warmtransfer.bench.analysis import recall_by_popularity_bucket
from warmtransfer.bench.datasets import datasets
from warmtransfer.bench.splitters.pseudo_cold import PseudoColdSplitter
from warmtransfer.columns import Columns as C
from warmtransfer.methods import methods
from warmtransfer.similarity import content_similarity
from warmtransfer.types import Dataset, TransferInputs

METHODS = ["grouped_most_popular_pers", "linmap", "stacking_plus"]
SEED = 42
K = 10
N_BUCKETS = 4


def main() -> None:
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "ml-1m"
    ds = datasets.get(dataset_name)().load()

    split = PseudoColdSplitter(
        cold_frac=0.2, val_frac=0.1, n_pop_buckets=5, min_item_interactions=5
    ).split(ds, SEED)
    assert ds.item_features is not None
    warm_features = ds.item_features.subset(split.warm_items)
    cold_features = ds.item_features.subset(split.cold_items)
    similarity = content_similarity(cold_features, warm_features)

    # популярность cold-айтемов по ПОЛНЫМ взаимодействиям (как в стратификации сплита)
    full_pop = cast("dict", ds.interactions.groupby(C.Item).size().to_dict())
    item_pop = {it: int(full_pop.get(it, 0)) for it in split.cold_items}

    # eval-пользователи и ground truth
    test_users = np.sort(np.asarray(split.test[C.User].unique()))
    if len(test_users) > 2000:
        rng = np.random.default_rng(SEED)
        test_users = np.sort(rng.choice(test_users, 2000, replace=False))
    gt = cast("pd.DataFrame", split.test[split.test[C.User].isin(test_users.tolist())])

    donor = ALSAdapter(factors=64, regularization=0.05, iterations=20)
    donor.fit(Dataset(split.train, warm_features, name=dataset_name), seed=SEED)
    donor_scores = donor.score(test_users, split.warm_items)

    val_cold_features = val_similarity = val_interactions = None
    if len(split.val):
        val_items = np.sort(np.asarray(split.val[C.Item].unique()))
        val_cold_features = ds.item_features.subset(val_items)
        val_similarity = content_similarity(val_cold_features, warm_features)
        val_interactions = cast(
            "pd.DataFrame", split.val[split.val[C.User].isin(test_users.tolist())]
        )

    inputs = TransferInputs(
        donor_scores=donor_scores,
        train_interactions=split.train,
        warm_features=warm_features,
        cold_features=cold_features,
        similarity=similarity,
        embeddings=donor.embeddings(),
        warm_items=split.warm_items,
        cold_items=split.cold_items,
        val_cold_features=val_cold_features,
        val_similarity=val_similarity,
        val_interactions=val_interactions,
    )

    print(f"\n=== recall@{K} по бакетам популярности cold-айтемов: {dataset_name} × als ===")
    print("(bucket 0 = самые нишевые cold-айтемы, выше = популярнее)\n")
    frames = []
    for name in METHODS:
        method = methods.get(name)().fit(inputs, seed=SEED)
        reco = method.predict(test_users, split.cold_items)
        tbl = recall_by_popularity_bucket(
            reco, gt, item_pop, n_buckets=N_BUCKETS, k=K
        )
        tbl.insert(0, "method", name)
        frames.append(tbl)

    result = pd.concat(frames, ignore_index=True)
    pivot = result.pivot(index="bucket", columns="method", values=f"recall@{K}")
    meta = cast(
        "pd.DataFrame",
        result[result["method"] == METHODS[0]][["bucket", "pop_range", "n_relevant"]],
    )
    merged = meta.merge(pivot.reset_index(), on="bucket")
    print(merged.to_markdown(index=False, floatfmt=".4f"))


if __name__ == "__main__":
    main()
