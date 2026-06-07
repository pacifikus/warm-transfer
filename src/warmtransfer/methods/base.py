"""Base cold-start method interface (score transfer/calibration) + registry.

This is the plug&play core: a method takes donor scores over warm items (+optionally
content/similarity/embeddings) and predicts scores for (user, cold_item) pairs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.exceptions import MissingInputError, NotFittedError
from warmtransfer.registry import Registry
from warmtransfer.types import TransferInputs

#: Allowed input sources that a method may declare in ``requires``.
INPUT_KINDS = frozenset(
    {
        "donor_scores",
        "train_interactions",
        "content",
        "similarity",
        "embeddings",
        "item_meta",
        "val",  # val-cold fold (for supervised methods)
    }
)


class ColdStartMethod(ABC):
    """Abstract cold-start method.

    Subclasses declare:
      * ``name`` — the name used in the registry/config;
      * ``requires`` — which inputs are mandatory (validated in :meth:`fit`).

    Contract:
      * :meth:`fit` returns ``self``;
      * :meth:`predict` returns a DataFrame ``[user_id, item_id, score]``;
      * deterministic for a fixed ``seed``.
    """

    name: str = "base"
    requires: frozenset[str] = frozenset({"donor_scores"})

    def __init__(self) -> None:
        self._fitted = False

    # --- contract ---

    @abstractmethod
    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        """Method training implementation (provided by the subclass)."""

    @abstractmethod
    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        """Predict scores for all (user_ids × cold_item_ids) pairs.

        Returns a long-format DataFrame ``[user_id, item_id, score]``.
        """

    # --- public API with input validation ---

    def fit(self, inputs: TransferInputs, seed: int = 0) -> ColdStartMethod:
        """Validate the mandatory inputs and train the method."""
        self._validate_inputs(inputs)
        self._fit(inputs, seed)
        self._fitted = True
        return self

    def _validate_inputs(self, inputs: TransferInputs) -> None:
        unknown = self.requires - INPUT_KINDS
        if unknown:
            raise ValueError(f"{self.name}: unknown requirements {unknown}")
        present = _present_inputs(inputs)
        missing = self.requires - present
        if missing:
            raise MissingInputError(
                f"Method {self.name!r} requires {sorted(self.requires)}, "
                f"but the following are missing: {sorted(missing)}"
            )

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError(f"{self.name}: call fit() before predict()")

    def get_params(self) -> dict:
        """Method hyperparameters (for logging into artifacts)."""
        return {}


def _present_inputs(inputs: TransferInputs) -> set[str]:
    present: set[str] = set()
    if inputs.donor_scores is not None and len(inputs.donor_scores):
        present.add("donor_scores")
    if inputs.train_interactions is not None and len(inputs.train_interactions):
        present.add("train_interactions")
    if inputs.warm_features is not None or inputs.cold_features is not None:
        present.add("content")
    if inputs.similarity is not None:
        present.add("similarity")
    if inputs.embeddings is not None:
        present.add("embeddings")
    if inputs.item_meta is not None:
        present.add("item_meta")
    if inputs.val_interactions is not None and inputs.val_cold_features is not None:
        present.add("val")
    return present


def cross_join_frame(
    user_ids: np.ndarray, cold_item_ids: np.ndarray, scores: np.ndarray
) -> pd.DataFrame:
    """Utility: build a long-format DataFrame from a dense score matrix [n_users, n_items]."""
    n_u, n_i = len(user_ids), len(cold_item_ids)
    if scores.shape != (n_u, n_i):
        raise ValueError(f"scores.shape={scores.shape} != ({n_u}, {n_i})")
    return pd.DataFrame(
        {
            C.User: np.repeat(user_ids, n_i),
            C.Item: np.tile(cold_item_ids, n_u),
            C.Score: scores.reshape(-1),
        }
    )


#: Global method registry.
methods: Registry[type[ColdStartMethod]] = Registry("method")


_M = TypeVar("_M", bound=ColdStartMethod)


def register_method(name: str) -> Callable[[type[_M]], type[_M]]:
    """Decorator that registers a method under ``name`` (sets ``cls.name``).

    Preserves the concrete class type (does not collapse it to the base) so that pyright
    sees the subclass ``__init__`` signature.
    """

    def decorator(cls: type[_M]) -> type[_M]:
        cls.name = name
        methods.register(name)(cls)
        return cls

    return decorator
