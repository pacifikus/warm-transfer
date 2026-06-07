"""Internal data types (hot path): dataclasses + numpy/pandas, no pydantic.

Pydantic is used only at the boundaries (configs, validation of public input in
``schema.py``). Here we keep lightweight containers for exchange between components.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

#: Feature/similarity matrix: dense or sparse.
Matrix = np.ndarray | sparse.spmatrix


@dataclass
class ItemFeatures:
    """Content features of items, aligned with ``item_ids``.

    ``matrix[i]`` is the feature vector of item ``item_ids[i]`` (external id).
    """

    item_ids: np.ndarray  # external ids, shape [n_items]
    matrix: Matrix  # [n_items, n_features]
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.item_ids)
        if self.matrix.shape[0] != n:
            raise ValueError(f"matrix.shape[0]={self.matrix.shape[0]} != len(item_ids)={n}")
        # map external id -> row position
        self._pos: dict = {iid: i for i, iid in enumerate(self.item_ids)}

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_features(self) -> int:
        return int(self.matrix.shape[1])

    def subset(self, ids: np.ndarray) -> ItemFeatures:
        """Subset by external ids (order follows ``ids``)."""
        rows = [self._pos[i] for i in ids]
        return ItemFeatures(
            item_ids=np.asarray(ids),
            matrix=self.matrix[rows],  # type: ignore[index]  # np.ndarray and sparse both support it
            feature_names=self.feature_names,
        )

    def has(self, item_id: object) -> bool:
        return item_id in self._pos


@dataclass
class Dataset:
    """Dataset: interactions + item content.

    Interactions are a long-format DataFrame with ``Columns`` columns (User, Item,
    Weight, Datetime). ``item_features`` is optional (content may be delegated to
    the library user).
    """

    interactions: pd.DataFrame
    item_features: ItemFeatures | None = None
    name: str = "dataset"

    @property
    def all_items(self) -> np.ndarray:
        from warmtransfer.columns import Columns as C

        return np.asarray(self.interactions[C.Item].unique())


@dataclass
class SplitResult:
    """Result of an honest cold-start split.

    Invariant (checked by tests): ``cold_items`` do not appear in ``train`` or
    ``val``; their interactions are present only in ``test``.
    """

    train: pd.DataFrame  # warm interactions for training the donor
    val: pd.DataFrame  # warm/cold validation (for hyperparameter tuning)
    test: pd.DataFrame  # interactions of cold items (ground truth)
    warm_items: np.ndarray
    cold_items: np.ndarray

    def cold_in_train(self) -> set:
        """Set of cold items leaked into train (must be empty)."""
        from warmtransfer.columns import Columns as C

        return set(self.cold_items) & set(self.train[C.Item].unique())


@dataclass
class TransferInputs:
    """Everything a cold-start method receives as input (``ColdStartMethod.fit``).

    The bundle is assembled by the benchmark runner from ``Dataset``/``SplitResult``
    and the donor. When using the library directly, the user fills in the needed
    fields themselves (at minimum — ``donor_scores`` + one content/similarity source).
    """

    donor_scores: pd.DataFrame  # [User, Item, Score] over WARM items
    train_interactions: pd.DataFrame = field(
        default_factory=pd.DataFrame
    )  # for popularity baselines (Grouped MP)
    warm_features: ItemFeatures | None = None
    cold_features: ItemFeatures | None = None
    # similarity: [n_cold, n_warm], aligned with *_features.item_ids
    similarity: np.ndarray | None = None
    # embeddings: {"warm_item": ..., "cold_item": ..., "user": ...}
    embeddings: dict[str, np.ndarray] | None = None
    warm_items: np.ndarray | None = None
    cold_items: np.ndarray | None = None
    item_meta: pd.DataFrame | None = None  # extra meta (item category) for Grouped MP

    # --- val-cold fold (for supervised methods: stacking, logreg calibration, tuning) ---
    # These items are also cold (absent from the donor train), but their interactions are
    # known (split.val), so a meta-model can be trained on them and predict on the test cold.
    val_cold_features: ItemFeatures | None = None
    val_similarity: np.ndarray | None = None  # [n_val_cold, n_warm]
    val_interactions: pd.DataFrame | None = None  # ground truth for val-cold
