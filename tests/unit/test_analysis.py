"""Analysis tests: recall by popularity buckets and RelaImpr in the table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.bench.analysis import recall_by_popularity_bucket
from warmtransfer.bench.results import add_rela_impr, to_table
from warmtransfer.columns import Columns as C


def test_recall_by_popularity_bucket() -> None:
    # 2 cold items: 100 (niche) and 200 (popular)
    item_pop = {100: 1, 200: 100}
    # user1: top-1 = item 200; both are relevant (100 and 200)
    reco = pd.DataFrame(
        {C.User: [1, 1], C.Item: [200, 100], C.Score: [0.9, 0.1]}
    )
    gt = pd.DataFrame({C.User: [1, 1], C.Item: [100, 200]})

    tbl = recall_by_popularity_bucket(reco, gt, item_pop, n_buckets=2, k=1)
    # bucket 0 (niche, item100): not in top-1 → recall 0
    # bucket 1 (popular, item200): in top-1 → recall 1
    by_bucket = dict(zip(tbl["bucket"], tbl["recall@1"], strict=True))
    assert by_bucket[0] == 0.0
    assert by_bucket[1] == 1.0


def test_add_rela_impr() -> None:
    records = [
        {"dataset": "d", "donor": "als", "method": "base", "seed": 0, "auc": 0.6},
        {"dataset": "d", "donor": "als", "method": "model", "seed": 0, "auc": 0.7},
    ]
    table = add_rela_impr(to_table(records), base_method="base")
    rela = dict(zip(table["method"], table["rela_impr"], strict=True))
    assert rela["base"] == 0.0
    # (0.7-0.5)/(0.6-0.5) - 1 = 2.0 - 1 = 1.0
    assert abs(rela["model"] - 1.0) < 1e-9


def test_add_rela_impr_missing_base() -> None:
    records = [{"dataset": "d", "donor": "als", "method": "model", "seed": 0, "auc": 0.7}]
    table = add_rela_impr(to_table(records), base_method="absent")
    assert np.isnan(table["rela_impr"].to_numpy()[0])
