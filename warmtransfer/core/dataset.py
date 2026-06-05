"""
Dataset — abstract interface, a generic processed-CSV loader, and a registry.

Two ways to add a dataset
-------------------------
1. STANDARD case (data already in the processed CSV format) — one line, no class:

       register("ml1m", ProcessedDataset("MovieLens-1M", "data/processed"))

2. CUSTOM case (the dataset needs genuinely different loading logic, e.g. a
   join on the fly or a non-CSV source) — subclass Dataset:

       @register_dataset("weird")
       class WeirdDataset(Dataset):
           name = "Weird"
           def load(self):
               ...
               return interactions_df, item_features_df

The per-dataset RAW preparation (parsing ratings.dat, building TF-IDF, ...) lives
in separate prep scripts under datasets/ — that is where real per-dataset
differences belong. Loading the prepared CSVs is uniform, so it is handled once
by ProcessedDataset rather than copy-pasted per dataset.

Standard processed schema (enforced by validate_schema)
-------------------------------------------------------
    interactions.csv  : must contain  user_id, item_id, engagement
    item_features.csv : must contain  item_id  + >=1 numeric feature column
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Union

import pandas as pd
from pandas.api.types import is_numeric_dtype


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------

REQUIRED_INTERACTION_COLS = ("user_id", "item_id", "engagement")


def validate_schema(
    interactions: pd.DataFrame,
    item_features: pd.DataFrame,
    name: str = "dataset",
) -> None:
    """
    Fail loudly if a dataset does not conform to the standard schema.

    This is what lets us add datasets 3, 4, 5 safely: a malformed prep output
    raises a clear error here instead of silently corrupting results downstream.
    """
    missing = [c for c in REQUIRED_INTERACTION_COLS if c not in interactions.columns]
    if missing:
        raise ValueError(
            f"[{name}] interactions is missing required columns {missing}. "
            f"Found: {list(interactions.columns)}"
        )

    if "item_id" not in item_features.columns:
        raise ValueError(f"[{name}] item_features must contain an 'item_id' column.")

    feature_cols = [c for c in item_features.columns if c != "item_id"]
    if not feature_cols:
        raise ValueError(
            f"[{name}] item_features has no feature columns (only 'item_id')."
        )

    non_numeric = [c for c in feature_cols if not is_numeric_dtype(item_features[c])]
    if non_numeric:
        raise ValueError(
            f"[{name}] item_features columns must be numeric; non-numeric: "
            f"{non_numeric[:10]}{' ...' if len(non_numeric) > 10 else ''}"
        )

    if item_features["item_id"].duplicated().any():
        n_dup = int(item_features["item_id"].duplicated().sum())
        raise ValueError(f"[{name}] item_features has {n_dup} duplicate item_id rows.")

    if len(interactions) == 0:
        raise ValueError(f"[{name}] interactions is empty.")

    # Soft check: how many interacted items actually have features (warn only)
    interacted = set(interactions["item_id"].unique())
    have_feats = set(item_features["item_id"].unique())
    missing_feats = len(interacted - have_feats)
    if missing_feats:
        frac = missing_feats / max(1, len(interacted))
        print(
            f"  [{name}] warning: {missing_feats:,} of {len(interacted):,} "
            f"interacted items ({frac:.1%}) have no features."
        )


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class Dataset(ABC):
    """
    Abstract dataset loader.

    load() must return (interactions_df, item_features_df) conforming to the
    standard schema. Concrete implementations should call validate_schema()
    before returning.
    """

    name: str = "unnamed"

    @abstractmethod
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (interactions_df, item_features_df)."""


# ---------------------------------------------------------------------------
# Generic concrete loader for the standard processed CSV pair
# ---------------------------------------------------------------------------

class ProcessedDataset(Dataset):
    """
    Loads a dataset already preprocessed into the standard two-CSV format.

    Covers the common case so individual datasets need no bespoke loader class.

    Parameters
    ----------
    name              : human-readable name (shown in result tables)
    data_dir          : directory holding the two CSVs
    interactions_file : interactions filename (default 'interactions.csv')
    features_file     : item-features filename (default 'item_features.csv')
    """

    def __init__(
        self,
        name: str,
        data_dir: str,
        interactions_file: str = "interactions.csv",
        features_file: str = "item_features.csv",
    ) -> None:
        self.name = name
        self.data_dir = data_dir
        self.interactions_file = interactions_file
        self.features_file = features_file

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ix_path = os.path.join(self.data_dir, self.interactions_file)
        fx_path = os.path.join(self.data_dir, self.features_file)
        for p in (ix_path, fx_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"[{self.name}] expected file not found: {p}. "
                    "Run the dataset's prep script first."
                )
        interactions = pd.read_csv(ix_path)
        item_features = pd.read_csv(fx_path)
        validate_schema(interactions, item_features, name=self.name)
        return interactions, item_features


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# A registry entry is either a ready Dataset instance or a Dataset subclass.
_REGISTRY: dict[str, Union[Dataset, type[Dataset]]] = {}


def register(key: str, dataset: Union[Dataset, type[Dataset]]) -> None:
    """
    Register a dataset under a short name.

    Pass either a ready instance (the common case):
        register("ml1m", ProcessedDataset("MovieLens-1M", "data/processed"))
    or a Dataset subclass (instantiated on demand by get_dataset).
    """
    _REGISTRY[key] = dataset


def register_dataset(key: str):
    """Decorator form of register() for custom Dataset subclasses."""
    def decorator(cls: type[Dataset]) -> type[Dataset]:
        _REGISTRY[key] = cls
        return cls
    return decorator


def get_dataset(key: str, **kwargs) -> Dataset:
    """
    Resolve a registered dataset by name.

    If the entry is an instance, it is returned as-is (kwargs ignored).
    If it is a subclass, it is instantiated with kwargs.
    """
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset '{key}'. Available: {sorted(_REGISTRY)}. "
            "Register new datasets with register() or @register_dataset."
        )
    entry = _REGISTRY[key]
    if isinstance(entry, Dataset):
        return entry
    return entry(**kwargs)


def list_datasets() -> list[str]:
    """Return names of all registered datasets."""
    return sorted(_REGISTRY.keys())
