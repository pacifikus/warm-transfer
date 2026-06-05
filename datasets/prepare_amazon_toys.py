"""
Preprocessing script for Amazon Toys & Games dataset (2018 version).

Input files (place in data/raw/amazon_toys/):
  - Toys_and_Games_5.json   : reviews (5-core, one JSON object per line)
  - meta_Toys_and_Games.json: item metadata (one JSON object per line)

Output (data/processed/amazon_toys/):
  - interactions.csv  : user_id, item_id, engagement, timestamp
  - item_features.csv : item_id + rich content features (see below)

Item features
-------------
We pull every signal the dataset gives us into a single feature vector,
so that content-based similarity can distinguish products that share a
category path but are otherwise very different (e.g. two LEGO sets in
"Building Sets" — same category, very different titles/descriptions).

  - cat_*    : multi-hot encoding of TOP_N_CATS most frequent sub-categories
               ("Toys & Games" itself is excluded — applies to everything)
  - brand_*  : one-hot encoding of TOP_N_BRANDS most frequent brands
  - title_*  : TF-IDF over product titles (N_TITLE_TFIDF terms)
  - text_*   : TF-IDF over product description + feature bullets (N_TEXT_TFIDF terms)
  - price_norm: min-max normalised price (NaN → -1 sentinel for "unknown")

Library design note
-------------------
The library itself (similarity / calibrator / aggregator) is feature-agnostic:
it only consumes a DataFrame [item_id, <any numeric columns>]. Dataset-specific
feature engineering lives here, in the dataset preprocessing script.

Run from repo root:
    python scripts/prepare_amazon_toys.py
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RAW_DIR        = "data/raw/amazon_toys"
OUT_DIR        = "data/processed/amazon_toys"
REVIEWS_FILE   = os.path.join(RAW_DIR, "Toys_and_Games_5.json")
META_FILE      = os.path.join(RAW_DIR, "meta_Toys_and_Games.json")

TOP_N_CATS     = 100   # keep top N most frequent sub-categories
MIN_CAT_FREQ   = 20    # cat must appear on ≥ this many items
TOP_N_BRANDS   = 100   # keep top N most frequent brands
MIN_BRAND_FREQ = 30    # brand must appear on ≥ this many items
N_TITLE_TFIDF  = 200   # TF-IDF dim for titles
N_TEXT_TFIDF   = 200   # TF-IDF dim for description + feature bullets
RATING_THRESH  = 4.0   # rating ≥ threshold → engagement = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_json_lines(path: str):
    """Yield parsed JSON objects from a file where each line is one object."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def parse_price(raw: str) -> float:
    """Extract numeric price from '$24.95'-style strings. NaN if unparseable."""
    if not isinstance(raw, str):
        return float("nan")
    raw = raw.strip()
    m = re.match(r"^\$(\d[\d,]*\.?\d*)", raw)
    if m:
        return float(m.group(1).replace(",", ""))
    return float("nan")


_HTML_RE  = re.compile(r"<[^>]+>")
_NONWORD_RE = re.compile(r"[^a-zA-Z0-9\s]")
_WS_RE    = re.compile(r"\s+")

def clean_text(s: str) -> str:
    """Lower-case, strip HTML tags, drop punctuation, collapse whitespace."""
    if not isinstance(s, str):
        return ""
    s = _HTML_RE.sub(" ", s)
    s = _NONWORD_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.lower().strip()


def safe_col(prefix: str, name: str, maxlen: int = 40) -> str:
    """Build a safe DataFrame column name."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(name)).strip("_")
    return f"{prefix}_{s[:maxlen]}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_reviews(path: str) -> pd.DataFrame:
    print(f"  Loading reviews from {path} ...")
    records = []
    for obj in tqdm(iter_json_lines(path), desc="  reviews"):
        records.append({
            "user_id":   obj.get("reviewerID"),
            "item_id":   obj.get("asin"),
            "rating":    float(obj.get("overall", 0)),
            "timestamp": int(obj.get("unixReviewTime", 0)),
        })
    df = pd.DataFrame(records).dropna(subset=["user_id", "item_id"])
    df["engagement"] = (df["rating"] >= RATING_THRESH).astype(int)
    print(f"  {len(df):,} reviews | "
          f"{df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")
    print(f"  Positive engagement rate: {df['engagement'].mean():.1%}")
    return df[["user_id", "item_id", "engagement", "timestamp"]]


def load_metadata(path: str, valid_item_ids: set) -> pd.DataFrame:
    print(f"\n  Loading metadata from {path} ...")
    rows = []
    for obj in tqdm(iter_json_lines(path), desc="  metadata"):
        asin = obj.get("asin")
        if asin not in valid_item_ids:
            continue
        desc_parts = obj.get("description") or []
        feat_parts = obj.get("feature") or []
        rows.append({
            "item_id":    asin,
            "categories": obj.get("category") or [],
            "brand":      (obj.get("brand") or "").strip(),
            "title":      obj.get("title") or "",
            "text":       " ".join(list(desc_parts) + list(feat_parts)),
            "price_raw":  obj.get("price") or "",
        })
    df = pd.DataFrame(rows).drop_duplicates("item_id").reset_index(drop=True)
    print(f"  Metadata for {len(df):,} / {len(valid_item_ids):,} items")
    return df


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_category_features(meta_df: pd.DataFrame) -> pd.DataFrame:
    print("\n  Categories ...")
    cnt: Counter = Counter()
    for cats in meta_df["categories"]:
        sub = [c for c in cats if c and c != "Toys & Games"]
        cnt.update(set(sub))
    top = [c for c, n in cnt.most_common(TOP_N_CATS) if n >= MIN_CAT_FREQ]
    print(f"    {len(cnt):,} unique sub-categories → keeping top {len(top)} "
          f"(min freq={MIN_CAT_FREQ})")

    cat_to_col = {c: i for i, c in enumerate(top)}
    M = np.zeros((len(meta_df), len(top)), dtype=np.float32)
    for i, cats in enumerate(meta_df["categories"]):
        for c in cats:
            j = cat_to_col.get(c)
            if j is not None:
                M[i, j] = 1.0
    covered = (M.sum(axis=1) > 0).sum()
    print(f"    Items with ≥1 category feature: {covered:,} / {len(meta_df):,}")
    cols = [safe_col("cat", c) for c in top]
    return pd.DataFrame(M, columns=cols)


def build_brand_features(meta_df: pd.DataFrame) -> pd.DataFrame:
    print("\n  Brands ...")
    counts = meta_df["brand"].value_counts()
    counts = counts[counts.index.astype(str).str.len() > 0]  # drop empty
    top = counts[counts >= MIN_BRAND_FREQ].head(TOP_N_BRANDS).index.tolist()
    print(f"    {(meta_df['brand'] != '').sum():,} items with brand | "
          f"{len(counts):,} unique brands → keeping top {len(top)} "
          f"(min freq={MIN_BRAND_FREQ})")

    brand_to_col = {b: i for i, b in enumerate(top)}
    M = np.zeros((len(meta_df), len(top)), dtype=np.float32)
    for i, b in enumerate(meta_df["brand"]):
        j = brand_to_col.get(b)
        if j is not None:
            M[i, j] = 1.0
    covered = (M.sum(axis=1) > 0).sum()
    print(f"    Items with brand feature: {covered:,} / {len(meta_df):,}")
    cols = [safe_col("brand", b) for b in top]
    return pd.DataFrame(M, columns=cols)


def build_tfidf_features(
    texts: list[str],
    n_features: int,
    prefix: str,
) -> pd.DataFrame:
    print(f"\n  TF-IDF over {prefix} (target {n_features} dims) ...")
    cleaned = [clean_text(t) for t in texts]
    non_empty = sum(1 for t in cleaned if t)
    print(f"    Non-empty {prefix}s: {non_empty:,} / {len(cleaned):,}")

    try:
        vec = TfidfVectorizer(
            max_features=n_features,
            min_df=10,
            stop_words="english",
            ngram_range=(1, 1),
        )
        X = vec.fit_transform(cleaned)
    except ValueError:
        # Vocabulary empty (all docs empty / stop-words only)
        print(f"    ⚠ TF-IDF produced empty vocabulary for {prefix}")
        return pd.DataFrame(np.zeros((len(texts), 0), dtype=np.float32))

    terms = vec.get_feature_names_out()
    print(f"    {len(terms)} terms extracted")
    X_arr = X.toarray().astype(np.float32)
    cols = [safe_col(prefix, t) for t in terms]
    return pd.DataFrame(X_arr, columns=cols)


def build_price_feature(meta_df: pd.DataFrame) -> pd.DataFrame:
    print("\n  Price ...")
    prices = meta_df["price_raw"].map(parse_price).values
    valid_mask = np.isfinite(prices)
    n_valid = int(valid_mask.sum())
    print(f"    Valid prices: {n_valid:,} / {len(meta_df):,}")

    if n_valid > 0:
        p_min = float(np.nanmin(prices))
        p_max = float(np.nanmax(prices))
        if p_max > p_min:
            norm = (prices - p_min) / (p_max - p_min)
        else:
            norm = np.zeros_like(prices)
    else:
        norm = np.full(len(prices), -1.0)
    norm = np.where(np.isfinite(norm), norm, -1.0).astype(np.float32)
    return pd.DataFrame({"price_norm": norm})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sep = "=" * 65
    print(sep)
    print("  Amazon Toys & Games — preprocessing (rich features)")
    print(sep)

    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Reviews
    print("\n[1/4] Reviews ...")
    interactions = load_reviews(REVIEWS_FILE)
    valid_items = set(interactions["item_id"].unique())

    # 2. Metadata
    print("\n[2/4] Metadata ...")
    meta = load_metadata(META_FILE, valid_items)

    # Keep only interactions for items that have metadata
    items_with_meta = set(meta["item_id"])
    before = interactions["item_id"].nunique()
    interactions = interactions[interactions["item_id"].isin(items_with_meta)]
    after = interactions["item_id"].nunique()
    if before != after:
        print(f"  Dropped {before - after:,} items with no metadata "
              f"({after:,} remaining)")

    # 3. Features
    print("\n[3/4] Building features ...")
    cat_df   = build_category_features(meta)
    brand_df = build_brand_features(meta)
    title_df = build_tfidf_features(meta["title"].tolist(),  N_TITLE_TFIDF, "title")
    text_df  = build_tfidf_features(meta["text"].tolist(),   N_TEXT_TFIDF,  "text")
    price_df = build_price_feature(meta)

    id_df = pd.DataFrame({"item_id": meta["item_id"].values})
    feat_df = pd.concat([id_df, cat_df, brand_df, title_df, text_df, price_df], axis=1)

    print(f"\n  Final feature matrix: {feat_df.shape}")
    print(f"    cat   : {cat_df.shape[1]}")
    print(f"    brand : {brand_df.shape[1]}")
    print(f"    title : {title_df.shape[1]}")
    print(f"    text  : {text_df.shape[1]}")
    print(f"    price : {price_df.shape[1]}")

    # 4. Save
    print("\n[4/4] Saving ...")
    interactions_path = os.path.join(OUT_DIR, "interactions.csv")
    features_path     = os.path.join(OUT_DIR, "item_features.csv")
    interactions.to_csv(interactions_path, index=False)
    feat_df.to_csv(features_path, index=False)

    print(f"\n  interactions.csv : {len(interactions):,} rows "
          f"({interactions['user_id'].nunique():,} users, "
          f"{interactions['item_id'].nunique():,} items)")
    print(f"  item_features.csv: {len(feat_df):,} items, "
          f"{len(feat_df.columns) - 1} feature columns")
    print(f"\n  Saved to {OUT_DIR}/")
    print(sep)


if __name__ == "__main__":
    main()
