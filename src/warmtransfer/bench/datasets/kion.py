"""MTS KION loader (movies/series, RU domain, implicit feedback).

Item content: multi-hot of genres + multi-hot of top-N countries + one-hot of the release
decade + one-hot of the content type (film/series). Interactions are implicit; we drop
accidental clicks using the ``watched_pct`` threshold and take a binary weight.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from warmtransfer.bench.datasets._download import cache_dir, download, unzip
from warmtransfer.bench.datasets.base import DatasetLoader, register_dataset
from warmtransfer.columns import Columns as C
from warmtransfer.types import Dataset, ItemFeatures

#: Mirror of the official dataset (English-language version of the content).
KION_URL = (
    "https://github.com/irsafilo/KION_DATASET/raw/"
    "f69775be31fa5779907cf0a92ddedb70037fb5ae/data_en.zip"
)

#: Minimum watch percentage below which an interaction is treated as an accidental click.
MIN_WATCHED_PCT = 10.0
#: How many of the most frequent countries to keep in the multi-hot.
TOP_COUNTRIES = 50
#: TF-IDF vocabulary size for the text content variant.
TFIDF_MAX_FEATURES = 1000


@register_dataset("kion")
class Kion(DatasetLoader):
    """MTS KION: ~860k users, ~15k items, ~5M views (implicit)."""

    def load(self) -> Dataset:
        interactions, items = _load_raw()
        return Dataset(
            interactions=interactions,
            item_features=_items_to_features(items),
            name="kion",
        )

    def describe(self) -> dict:
        return {
            "name": "kion",
            "domain": "movies/series (RU)",
            "feedback": "implicit (watched_pct)",
            "size": "~860k users, ~15k items, ~5M views",
            "content": (
                f"multi-hot of genres + multi-hot of top-{TOP_COUNTRIES} countries + "
                "one-hot of decade + one-hot of type (film/series)"
            ),
            "url": KION_URL,
        }


@register_dataset("kion-text")
class KionText(DatasetLoader):
    """KION with text content: TF-IDF over description/keywords/genres/title ([MA]).

    The same interactions as ``kion``, but the item content is a TF-IDF text vector (rather
    than categorical one-hot). Lets us check whether rich text content improves score
    transfer to cold items.
    """

    def load(self) -> Dataset:
        interactions, items = _load_raw()
        return Dataset(
            interactions=interactions,
            item_features=_items_to_tfidf(items),
            name="kion-text",
        )

    def describe(self) -> dict:
        return {
            "name": "kion-text",
            "domain": "movies/series (RU)",
            "feedback": "implicit (watched_pct)",
            "content": (
                f"TF-IDF (max_features={TFIDF_MAX_FEATURES}) over "
                "description + keywords + genres + title"
            ),
            "url": KION_URL,
        }


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download KION and return (interactions long-format, items DataFrame)."""
    root = cache_dir("kion")
    archive = download(KION_URL, root / "data_en.zip")
    data_dir = unzip(archive, root / "extracted") / "data_en"

    inter = pd.read_csv(data_dir / "interactions.csv")
    watched = cast("pd.Series", inter["watched_pct"]).fillna(0.0)
    inter = cast("pd.DataFrame", inter[watched >= MIN_WATCHED_PCT]).copy()

    dt = pd.to_datetime(cast("pd.Series", inter["last_watch_dt"]), errors="coerce")
    interactions = pd.DataFrame(
        {
            C.User: cast("pd.Series", inter["user_id"]).to_numpy(),
            C.Item: cast("pd.Series", inter["item_id"]).to_numpy(),
            C.Weight: 1.0,
            C.Datetime: dt.astype("int64").to_numpy(),
        }
    )
    items = pd.read_csv(data_dir / "items_en.csv", index_col=0)
    return interactions, items


def _items_to_tfidf(items: pd.DataFrame) -> ItemFeatures:
    """TF-IDF over the combined text (description + keywords + genres + title)."""
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer

    item_ids = np.asarray(cast("pd.Series", items["item_id"]).to_numpy())

    def _text(col: str) -> list[str]:
        return [s if isinstance(s, str) else "" for s in cast("pd.Series", items[col]).tolist()]

    parts = [_text(c) for c in ("title", "genres", "keywords", "description")]
    corpus = [" ".join(fields) for fields in zip(*parts, strict=True)]

    vec = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words="english")
    tfidf = csr_matrix(vec.fit_transform(corpus))
    # densify: several methods expect a dense np.ndarray (np.asarray(..., dtype=float))
    matrix = np.asarray(tfidf.todense(), dtype=np.float32)
    feature_names = [f"tfidf={t}" for t in vec.get_feature_names_out().tolist()]
    return ItemFeatures(item_ids=item_ids, matrix=matrix, feature_names=feature_names)


def _split_list(value: object) -> list[str]:
    """Split a string like 'drama, foreign' into a list of normalized tokens."""
    if not isinstance(value, str):
        return []
    return [t.strip().lower() for t in value.split(",") if t.strip()]


def _decade_bucket(year: object) -> str:
    if year is None or (isinstance(year, float) and np.isnan(year)):
        return "decade_unknown"
    try:
        y = int(float(cast("float", year)))
    except (ValueError, TypeError):
        return "decade_unknown"
    return f"decade_{y // 10 * 10}"


def _items_to_features(items: pd.DataFrame) -> ItemFeatures:
    """KION content -> float matrix (genres + countries + decade + content type)."""
    item_ids = np.asarray(cast("pd.Series", items["item_id"]).to_numpy())
    n = len(items)

    genres = [_split_list(g) for g in cast("pd.Series", items["genres"]).tolist()]
    countries = [_split_list(c) for c in cast("pd.Series", items["countries"]).tolist()]
    decades = [_decade_bucket(y) for y in cast("pd.Series", items["release_year"]).tolist()]
    ctypes = [
        str(c).strip().lower() if isinstance(c, str) else "type_unknown"
        for c in cast("pd.Series", items["content_type"]).tolist()
    ]

    # feature vocabularies
    all_genres = sorted({g for row in genres for g in row})
    top_countries = [
        str(c)
        for c in cast(
            "list[str]",
            pd.Series([c for row in countries for c in row]).value_counts().index.tolist(),
        )[:TOP_COUNTRIES]
    ]
    all_decades = sorted(set(decades))
    all_ctypes = sorted(set(ctypes))

    g_idx = {g: j for j, g in enumerate(all_genres)}
    c_idx = {c: j for j, c in enumerate(top_countries)}
    d_idx = {d: j for j, d in enumerate(all_decades)}
    t_idx = {t: j for j, t in enumerate(all_ctypes)}

    n_g, n_c, n_d, n_t = len(g_idx), len(c_idx), len(d_idx), len(t_idx)
    off_c, off_d, off_t = n_g, n_g + n_c, n_g + n_c + n_d
    matrix = np.zeros((n, n_g + n_c + n_d + n_t), dtype=np.float32)

    for i in range(n):
        for g in genres[i]:
            matrix[i, g_idx[g]] = 1.0
        for c in countries[i]:
            if c in c_idx:
                matrix[i, off_c + c_idx[c]] = 1.0
        matrix[i, off_d + d_idx[decades[i]]] = 1.0
        matrix[i, off_t + t_idx[ctypes[i]]] = 1.0

    feature_names = (
        [f"genre={g}" for g in all_genres]
        + [f"country={c}" for c in top_countries]
        + list(all_decades)
        + [f"type={t}" for t in all_ctypes]
    )
    return ItemFeatures(item_ids=item_ids, matrix=matrix, feature_names=feature_names)
