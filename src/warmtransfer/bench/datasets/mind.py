"""MIND (Microsoft News Dataset) loader — news domain, strong text content.

Source — the official HuggingFace mirror ``yjw1029/MIND`` (also linked from
msnews.github.io; the old Azure blob is gone). We take the full MINDlarge: train + dev.
We skip Test — there the click labels of impressions are HIDDEN (the dataset is a
competition and the answers are held by the leaderboard server), so there is no ground
truth for offline metrics in it.

Files (tab-separated):
  * ``behaviors.tsv`` — ``ImpressionID, UserID, Time, History, Impressions``.
    ``Impressions`` is a list of candidates like ``N69938-1`` (clicked) / ``N91737-0``
    (not). Interactions = CLICKS (label ``-1``) with the impression timestamp.
  * ``news.tsv`` — ``NewsID, Category, SubCategory, Title, Abstract, URL, *Entities``.

News content: categorical (category + subcategory) → ``mind``; text (TF-IDF over
title + abstract) → ``mind-text``. The same "categories vs text" axis as
``amazon-toys`` / ``amazon-toys-text`` and ``kion`` / ``kion-text``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from warmtransfer.bench.datasets._download import cache_dir, download, unzip
from warmtransfer.bench.datasets.base import DatasetLoader, register_dataset
from warmtransfer.columns import Columns as C
from warmtransfer.types import Dataset, ItemFeatures

MIND_BASE = "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/"
#: Take the full (large) variant: train + dev. Test has no labels — not used.
MIND_SPLITS = ("MINDlarge_train", "MINDlarge_dev")

#: TF-IDF vocabulary size for the text content variant.
TFIDF_MAX_FEATURES = 1000
#: Timestamp format in behaviors.tsv ("11/15/2019 8:55:22 AM").
_TIME_FMT = "%m/%d/%Y %I:%M:%S %p"


@register_dataset("mind")
class Mind(DatasetLoader):
    """MIND (large): news, implicit clicks; content = category + subcategory (one-hot)."""

    def load(self) -> Dataset:
        interactions, news = _load_raw()
        interactions, item_features = _align(interactions, news, _news_to_features)
        return Dataset(interactions=interactions, item_features=item_features, name="mind")

    def describe(self) -> dict:
        return {
            "name": "mind",
            "domain": "news (EN)",
            "feedback": "implicit (clicks in impressions)",
            "content": "one-hot category + one-hot subcategory",
            "url": MIND_BASE,
        }


@register_dataset("mind-text")
class MindText(DatasetLoader):
    """MIND with TEXT content: TF-IDF over the news title + abstract.

    The same interactions as ``mind``, but the item content is a TF-IDF text vector.
    Lets us fairly compare the contribution of rich text vs simple categories to score
    transfer.
    """

    def load(self) -> Dataset:
        interactions, news = _load_raw()
        interactions, item_features = _align(interactions, news, _news_to_tfidf)
        return Dataset(interactions=interactions, item_features=item_features, name="mind-text")

    def describe(self) -> dict:
        return {
            "name": "mind-text",
            "domain": "news (EN)",
            "feedback": "implicit (clicks in impressions)",
            "content": f"TF-IDF (max_features={TFIDF_MAX_FEATURES}) over title + abstract",
            "url": MIND_BASE,
        }


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download MINDlarge (train+dev) and assemble (interactions long-format, news DataFrame)."""
    root = cache_dir("mind")
    inter_parts: list[pd.DataFrame] = []
    news_parts: list[pd.DataFrame] = []
    for name in MIND_SPLITS:
        archive = download(MIND_BASE + name + ".zip", root / f"{name}.zip")
        data_dir = unzip(archive, root / f"{name}_x") / name
        inter_parts.append(_clicks_from_behaviors(data_dir / "behaviors.tsv"))
        news_parts.append(_read_news(data_dir / "news.tsv"))

    interactions = pd.concat(inter_parts, ignore_index=True)
    interactions = interactions.drop_duplicates(subset=[C.User, C.Item]).reset_index(drop=True)
    news = pd.concat(news_parts, ignore_index=True).drop_duplicates(subset="news_id")
    return interactions, news


def _clicks_from_behaviors(path: Path) -> pd.DataFrame:
    """Extract CLICKS (label ``-1``) from behaviors.tsv as [User, Item, Weight, Datetime]."""
    beh = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["imp_id", "user", "time", "history", "impr"],
        usecols=["user", "time", "impr"],
        dtype=str,
        quoting=csv.QUOTE_NONE,
    )
    # only clicked candidates (ending in "-1"); a list of short tokens per impression
    clicked = cast("pd.Series", beh["impr"]).str.findall(r"(N\d+)-1")
    times = pd.to_datetime(cast("pd.Series", beh["time"]), format=_TIME_FMT, errors="coerce")
    df = pd.DataFrame({"user": beh["user"], "time": times, "item": clicked})
    df = df.explode("item")
    df = cast("pd.DataFrame", df[df["item"].notna() & df["time"].notna()])
    return pd.DataFrame(
        {
            C.User: df["user"].to_numpy(),
            C.Item: df["item"].to_numpy(),
            C.Weight: 1.0,
            C.Datetime: cast("pd.Series", df["time"]).astype("int64").to_numpy(),
        }
    )


def _read_news(path: Path) -> pd.DataFrame:
    """news.tsv → DataFrame [news_id, category, subcategory, title, abstract]."""
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["news_id", "category", "subcategory", "title", "abstract", "url", "te", "ae"],
        usecols=["news_id", "category", "subcategory", "title", "abstract"],
        dtype=str,
        quoting=csv.QUOTE_NONE,
    )


def _align(
    interactions: pd.DataFrame,
    news: pd.DataFrame,
    build_features,
) -> tuple[pd.DataFrame, ItemFeatures]:
    """Keep only items that have news metadata, and build features for them."""
    news_ids = set(cast("pd.Series", news["news_id"]).tolist())
    interactions = interactions[interactions[C.Item].isin(news_ids)].copy()
    kept_items = np.asarray(interactions[C.Item].unique())
    news_by_id = news.set_index("news_id")
    sub = news_by_id.loc[kept_items]
    return interactions, build_features(kept_items, sub)


def _news_to_features(item_ids: np.ndarray, news: pd.DataFrame) -> ItemFeatures:
    """One-hot category + one-hot subcategory (aligned to ``item_ids``)."""
    cats = [str(c) if isinstance(c, str) else "" for c in cast("pd.Series", news["category"])]
    subs = [str(s) if isinstance(s, str) else "" for s in cast("pd.Series", news["subcategory"])]

    all_cats = sorted(set(cats))
    all_subs = sorted(set(subs))
    cat_idx = {c: j for j, c in enumerate(all_cats)}
    sub_idx = {s: j for j, s in enumerate(all_subs)}

    n_cat = len(all_cats)
    matrix = np.zeros((len(item_ids), n_cat + len(all_subs)), dtype=np.float32)
    for i in range(len(item_ids)):
        matrix[i, cat_idx[cats[i]]] = 1.0
        matrix[i, n_cat + sub_idx[subs[i]]] = 1.0

    feature_names = [f"cat={c}" for c in all_cats] + [f"sub={s}" for s in all_subs]
    return ItemFeatures(item_ids=np.asarray(item_ids), matrix=matrix, feature_names=feature_names)


def _news_to_tfidf(item_ids: np.ndarray, news: pd.DataFrame) -> ItemFeatures:
    """TF-IDF over the combined text (title + abstract)."""
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer

    def _text(col: str) -> list[str]:
        return [s if isinstance(s, str) else "" for s in cast("pd.Series", news[col]).tolist()]

    titles, abstracts = _text("title"), _text("abstract")
    corpus = [f"{t} {a}".strip() for t, a in zip(titles, abstracts, strict=True)]

    vec = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words="english")
    tfidf = csr_matrix(vec.fit_transform(corpus))
    matrix = np.asarray(tfidf.todense(), dtype=np.float32)
    feature_names = [f"tfidf={t}" for t in vec.get_feature_names_out().tolist()]
    return ItemFeatures(item_ids=np.asarray(item_ids), matrix=matrix, feature_names=feature_names)
