"""Test building Amazon Toys content features from a synthetic meta dict (no download).

``_meta_to_features`` is the error-prone part: multi-hot of the top categories followed
by one-hot of the top brand, laid out in a single matrix with the right column offsets.
"""

from __future__ import annotations

import numpy as np

from warmtransfer.bench.datasets.amazon import _meta_to_features


def _meta() -> dict[str, dict]:
    return {
        "A": {"categories": ["Toys", "Games"], "brand": "Lego", "title": "", "description": ""},
        "B": {"categories": ["Toys"], "brand": "Hasbro", "title": "", "description": ""},
        "C": {"categories": ["Games", "Puzzles"], "brand": "Lego", "title": "", "description": ""},
    }


def test_meta_to_features_multihot_and_brand() -> None:
    item_ids = np.array(["A", "B", "C"])
    feats = _meta_to_features(item_ids, _meta())
    mat = np.asarray(feats.matrix)
    names = feats.feature_names

    assert list(feats.item_ids) == ["A", "B", "C"]
    assert mat.shape[0] == 3

    # categories — multi-hot: A has Toys AND Games
    toys = names.index("cat=Toys")
    games = names.index("cat=Games")
    assert mat[0, toys] == 1.0
    assert mat[0, games] == 1.0
    assert mat[1, games] == 0.0  # B only has Toys

    # brand — one-hot: A and C are Lego, B is Hasbro
    lego = names.index("brand=Lego")
    hasbro = names.index("brand=Hasbro")
    assert mat[0, lego] == 1.0
    assert mat[2, lego] == 1.0
    assert mat[1, hasbro] == 1.0
    assert mat[1, lego] == 0.0


def test_meta_to_features_one_brand_per_item() -> None:
    """Each item activates exactly one brand column."""
    item_ids = np.array(["A", "B", "C"])
    feats = _meta_to_features(item_ids, _meta())
    mat = np.asarray(feats.matrix)
    names = feats.feature_names
    brand_cols = [j for j, n in enumerate(names) if n.startswith("brand=")]
    assert mat[:, brand_cols].sum(axis=1).tolist() == [1.0, 1.0, 1.0]
