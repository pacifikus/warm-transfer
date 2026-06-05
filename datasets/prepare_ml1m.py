"""
Preprocesses MovieLens-1M dataset into standardised format for the warm-transfer benchmark.

Input  (data/ml-1m/):
    ratings.dat  — UserID::MovieID::Rating::Timestamp
    movies.dat   — MovieID::Title::Genres  (pipe-separated genres)

Output (data/processed/):
    interactions.csv   — [user_id, item_id, timestamp, engagement]
    item_features.csv  — [item_id, genre_*]  multi-hot binary genre vectors

Run:
    python prepare_data.py
"""

import os
import pandas as pd

DATA_DIR = "data/ml-1m"
OUT_DIR = "data/processed"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ratings() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "ratings.dat")
    df = pd.read_csv(
        path, sep="::", engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    # Binary engagement: positive rating (>= 4) = 1, otherwise = 0
    df["engagement"] = (df["rating"] >= 4).astype(int)
    df = df.drop(columns=["rating"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df[["user_id", "item_id", "timestamp", "engagement"]]


def load_movies() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "movies.dat")
    df = pd.read_csv(
        path, sep="::", engine="python",
        names=["item_id", "title", "genres"],
        encoding="latin-1",
    )
    return df


# ---------------------------------------------------------------------------
# Feature engineering: multi-hot genre vectors
# ---------------------------------------------------------------------------

def build_genre_features(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the pipe-separated genres column into binary indicator columns.

    Example:
        "Action|Adventure|Sci-Fi"  →  genre_Action=1, genre_Adventure=1, genre_SciFi=1
    """
    # Collect all unique genre labels
    all_genres: set[str] = set()
    for genres_str in movies_df["genres"].dropna():
        all_genres.update(genres_str.split("|"))
    all_genres = sorted(all_genres)

    # Build safe column names (no spaces, no apostrophes)
    def safe_col(g: str) -> str:
        return "genre_" + g.replace(" ", "_").replace("'", "").replace("-", "_")

    rows = []
    for _, row in movies_df.iterrows():
        genre_set = set(row["genres"].split("|")) if pd.notna(row["genres"]) else set()
        feature_row: dict = {"item_id": row["item_id"]}
        for g in all_genres:
            feature_row[safe_col(g)] = int(g in genre_set)
        rows.append(feature_row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Interactions ---
    print("Loading ratings.dat ...")
    interactions = load_ratings()
    out_path = os.path.join(OUT_DIR, "interactions.csv")
    interactions.to_csv(out_path, index=False)

    counts = interactions.groupby("item_id").size()
    print(f"  Saved {len(interactions):,} interactions → {out_path}")
    print(f"  Users : {interactions['user_id'].nunique():,}")
    print(f"  Items : {interactions['item_id'].nunique():,}")
    print(f"  Positive rate : {interactions['engagement'].mean():.2%}")
    print(f"  Interactions per item — "
          f"min {counts.min()}, median {counts.median():.0f}, "
          f"mean {counts.mean():.1f}, max {counts.max()}")

    print()
    for n in [10, 20, 50, 100, 200]:
        n_warm = int((counts >= n).sum())
        print(f"  Items with >= {n:>3} interactions: {n_warm:>4}  ({n_warm / len(counts):.1%})")

    # --- Item features ---
    print("\nLoading movies.dat and building genre features ...")
    movies = load_movies()
    item_features = build_genre_features(movies)
    genre_cols = [c for c in item_features.columns if c != "item_id"]

    out_path = os.path.join(OUT_DIR, "item_features.csv")
    item_features.to_csv(out_path, index=False)
    print(f"  Saved features for {len(item_features):,} items → {out_path}")
    print(f"  Genre dimensions: {len(genre_cols)}")
    print(f"  Genres: {', '.join(g.replace('genre_', '') for g in genre_cols)}")


if __name__ == "__main__":
    main()
