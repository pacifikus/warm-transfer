"""Test building MIND content features from a synthetic news frame (no download).

``_news_to_features`` produces one-hot of category followed by one-hot of subcategory,
aligned row-for-row to ``item_ids``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.bench.datasets.mind import _news_to_features


def test_news_to_features_one_hot() -> None:
    item_ids = np.array(["N1", "N2", "N3"])
    news = pd.DataFrame(
        {
            "category": ["sports", "news", "sports"],
            "subcategory": ["football", "politics", "basketball"],
        }
    )
    feats = _news_to_features(item_ids, news)
    mat = np.asarray(feats.matrix)
    names = feats.feature_names

    assert list(feats.item_ids) == ["N1", "N2", "N3"]
    assert mat.shape[0] == 3

    # category — one-hot: N1 and N3 are sports, N2 is news
    assert mat[0, names.index("cat=sports")] == 1.0
    assert mat[2, names.index("cat=sports")] == 1.0
    assert mat[1, names.index("cat=news")] == 1.0
    assert mat[1, names.index("cat=sports")] == 0.0

    # subcategory — one-hot
    assert mat[0, names.index("sub=football")] == 1.0
    assert mat[1, names.index("sub=politics")] == 1.0
    assert mat[2, names.index("sub=basketball")] == 1.0


def test_news_to_features_exactly_one_cat_and_sub_per_item() -> None:
    item_ids = np.array(["N1", "N2", "N3"])
    news = pd.DataFrame(
        {
            "category": ["sports", "news", "sports"],
            "subcategory": ["football", "politics", "basketball"],
        }
    )
    feats = _news_to_features(item_ids, news)
    mat = np.asarray(feats.matrix)
    names = feats.feature_names
    cat_cols = [j for j, n in enumerate(names) if n.startswith("cat=")]
    sub_cols = [j for j, n in enumerate(names) if n.startswith("sub=")]
    assert mat[:, cat_cols].sum(axis=1).tolist() == [1.0, 1.0, 1.0]
    assert mat[:, sub_cols].sum(axis=1).tolist() == [1.0, 1.0, 1.0]
