"""Amazon Product Reviews loader (Toys & Games category), 5-core (SNAP, 2014).

Source — Julian McAuley, SNAP 5-core dumps: every user and item has >=5 reviews.
Two files:
  * ``reviews_*_5.json.gz`` — reviews. Each line is valid JSON
    (``reviewerID``, ``asin``, ``overall``, ``unixReviewTime``).
  * ``meta_*.json.gz`` — item metadata. Each line is a python-dict literal
    (single quotes!), so it is parsed with ``ast.literal_eval``, NOT ``json``.

Item content here is CATEGORICAL: multi-hot of the top-N leaf categories + one-hot
of the top-N brands. Text content (title/description via TF-IDF) lives in a separate
loader ``amazon-toys-text`` — so we can fairly compare "does rich text help score
transfer vs simple categories" (a benchmark axis).
"""

from __future__ import annotations

import ast
import gzip
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from warmtransfer.bench.datasets._download import cache_dir, download
from warmtransfer.bench.datasets.base import DatasetLoader, register_dataset
from warmtransfer.columns import Columns as C
from warmtransfer.types import Dataset, ItemFeatures

TOYS_REVIEWS_URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
    "reviews_Toys_and_Games_5.json.gz"
)
TOYS_META_URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
    "meta_Toys_and_Games.json.gz"
)

#: How many of the most frequent categories/brands to keep in the one-hot.
TOP_CATEGORIES = 200
TOP_BRANDS = 200
#: TF-IDF vocabulary size for the text content variant.
TFIDF_MAX_FEATURES = 1000


@register_dataset("amazon-toys")
class AmazonToys(DatasetLoader):
    """Amazon Toys & Games (5-core): explicit ratings 1..5, content = categories + brand."""

    def load(self) -> Dataset:
        root = cache_dir("amazon-toys")
        reviews_path = download(TOYS_REVIEWS_URL, root / "reviews_Toys_and_Games_5.json.gz")
        meta_path = download(TOYS_META_URL, root / "meta_Toys_and_Games.json.gz")

        interactions = _load_interactions(reviews_path)
        meta = _load_meta(meta_path)

        # Coverage guarantee: keep only items that have metadata features.
        items_with_meta = set(meta.keys())
        interactions = interactions[interactions[C.Item].isin(items_with_meta)].copy()

        kept_items = np.asarray(interactions[C.Item].unique())
        item_features = _meta_to_features(kept_items, meta)
        return Dataset(interactions=interactions, item_features=item_features, name="amazon-toys")

    def describe(self) -> dict:
        return {
            "name": "amazon-toys",
            "domain": "products (toys and games)",
            "feedback": "explicit (ratings 1..5)",
            "content": (
                f"multi-hot of top-{TOP_CATEGORIES} categories + one-hot of top-{TOP_BRANDS} brands"
            ),
            "reviews_url": TOYS_REVIEWS_URL,
            "meta_url": TOYS_META_URL,
        }


@register_dataset("amazon-toys-text")
class AmazonToysText(DatasetLoader):
    """Amazon Toys & Games with TEXT content: TF-IDF over title + description.

    The same interactions as ``amazon-toys``, but the item content is a TF-IDF text
    vector (rather than categorical one-hot). The pair ``amazon-toys`` /
    ``amazon-toys-text`` lets us fairly compare whether rich text improves score transfer
    to cold items relative to simple categories (the same axis as ``kion`` / ``kion-text``).
    """

    def load(self) -> Dataset:
        root = cache_dir("amazon-toys")
        reviews_path = download(TOYS_REVIEWS_URL, root / "reviews_Toys_and_Games_5.json.gz")
        meta_path = download(TOYS_META_URL, root / "meta_Toys_and_Games.json.gz")

        interactions = _load_interactions(reviews_path)
        meta = _load_meta(meta_path)

        items_with_meta = set(meta.keys())
        interactions = interactions[interactions[C.Item].isin(items_with_meta)].copy()

        kept_items = np.asarray(interactions[C.Item].unique())
        item_features = _meta_to_tfidf(kept_items, meta)
        return Dataset(
            interactions=interactions, item_features=item_features, name="amazon-toys-text"
        )

    def describe(self) -> dict:
        return {
            "name": "amazon-toys-text",
            "domain": "products (toys and games)",
            "feedback": "explicit (ratings 1..5)",
            "content": f"TF-IDF (max_features={TFIDF_MAX_FEATURES}) over title + description",
            "reviews_url": TOYS_REVIEWS_URL,
            "meta_url": TOYS_META_URL,
        }


def _iter_json_lines(path: Path) -> Iterator[dict]:
    """Read a gzip with valid JSON per line (reviews), line by line."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_literal_lines(path: Path) -> Iterator[dict]:
    """Read a gzip with python-dict literals per line (metadata: single quotes)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield ast.literal_eval(line)


def _load_interactions(reviews_path: Path) -> pd.DataFrame:
    """Reviews → long format [User, Item, Weight, Datetime]."""
    users: list[str] = []
    items: list[str] = []
    weights: list[float] = []
    times: list[int] = []
    for r in _iter_json_lines(reviews_path):
        users.append(r["reviewerID"])
        items.append(r["asin"])
        weights.append(float(r.get("overall", 0.0)))
        times.append(int(r.get("unixReviewTime", 0)))
    return pd.DataFrame(
        {
            C.User: users,
            C.Item: items,
            C.Weight: weights,
            C.Datetime: times,
        }
    )


def _load_meta(meta_path: Path) -> dict[str, dict]:
    """Metadata → {asin: {"categories": [str, ...], "brand": str}}."""
    meta: dict[str, dict] = {}
    for m in _iter_literal_lines(meta_path):
        asin = m.get("asin")
        if asin is None:
            continue
        # categories is a list of lists (paths down the category tree); flatten to a set.
        cats_raw = m.get("categories") or []
        cats: set[str] = set()
        for path in cats_raw:
            for c in path:
                cats.add(str(c))
        brand = m.get("brand")
        title = m.get("title")
        desc = m.get("description")
        meta[asin] = {
            "categories": sorted(cats),
            "brand": str(brand).strip() if isinstance(brand, str) and brand.strip() else "",
            "title": title if isinstance(title, str) else "",
            "description": desc if isinstance(desc, str) else "",
        }
    return meta


def _top_n(counter: pd.Series, n: int) -> dict[str, int]:
    """Top-N values by frequency → {value: column index}."""
    ranked = [str(v) for v in counter.index.tolist()[:n]]
    return {v: j for j, v in enumerate(ranked)}


def _meta_to_features(item_ids: np.ndarray, meta: dict[str, dict]) -> ItemFeatures:
    """Multi-hot of top-N categories + one-hot of top-N brands in one float matrix.

    ``feature_names`` — categories first (prefix ``cat=``), then brands (``brand=``).
    """
    n = len(item_ids)

    # Frequencies over the kept items.
    cat_counts = pd.Series(
        [c for iid in item_ids for c in meta[iid]["categories"]]
    ).value_counts()
    brand_counts = pd.Series(
        [meta[iid]["brand"] for iid in item_ids if meta[iid]["brand"]]
    ).value_counts()

    cat_idx = _top_n(cat_counts, TOP_CATEGORIES)
    brand_idx = _top_n(brand_counts, TOP_BRANDS)

    n_cat = len(cat_idx)
    n_brand = len(brand_idx)
    matrix = np.zeros((n, n_cat + n_brand), dtype=np.float32)

    for i, iid in enumerate(item_ids):
        for c in meta[iid]["categories"]:
            j = cat_idx.get(c)
            if j is not None:
                matrix[i, j] = 1.0
        b = meta[iid]["brand"]
        bj = brand_idx.get(b)
        if bj is not None:
            matrix[i, n_cat + bj] = 1.0

    feature_names = [f"cat={c}" for c in cat_idx] + [f"brand={b}" for b in brand_idx]
    return ItemFeatures(item_ids=item_ids, matrix=matrix, feature_names=feature_names)


def _meta_to_tfidf(item_ids: np.ndarray, meta: dict[str, dict]) -> ItemFeatures:
    """TF-IDF over the combined text (title + description)."""
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = [
        f"{meta[iid]['title']} {meta[iid]['description']}".strip() for iid in item_ids
    ]
    vec = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words="english")
    tfidf = csr_matrix(vec.fit_transform(corpus))
    # densify: several methods expect a dense np.ndarray
    matrix = np.asarray(tfidf.todense(), dtype=np.float32)
    feature_names = [f"tfidf={t}" for t in vec.get_feature_names_out().tolist()]
    return ItemFeatures(item_ids=np.asarray(item_ids), matrix=matrix, feature_names=feature_names)
