"""
Model — abstract base class for recommendation-model wrappers.

Any model passed to warmtransfer must implement this interface. The model is
treated as a *black box*: warmtransfer never retrains it for the cold-start task.
It only calls predict_batch (and, for embedding-based methods, the embedding
accessors).

Design note — why hyperparameters live in __init__, not fit
-----------------------------------------------------------
A concrete model carries its own training config (factors, iterations,
min_user_interactions, ...) in its constructor. That keeps fit(warm_df)
uniform across all model families, so the runner never has to know
model-specific knobs:

    model_factory = lambda: BPRModel(factors=64, iterations=200, ...)
    model = model_factory()
    model.fit(warm_df)          # same call for BPR, ALS, Two-Tower, ...

Required interface
------------------
    fit(warm_df) -> self
    predict_batch(user_ids, item_ids) -> np.ndarray

Optional (only for embedding-based cold-start methods)
------------------------------------------------------
    supports_embeddings -> bool
    get_item_embeddings_batch(item_ids) -> np.ndarray
    get_user_embeddings_batch(user_ids) -> np.ndarray

Optional (only for honest generalisation validation)
----------------------------------------------------
    known_user_ids -> list
    known_item_ids -> list
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Model(ABC):
    """
    Abstract wrapper around a trained recommendation model.

    Concrete implementation: warmtransfer.models.bpr.BPRModel
    """

    # Human-readable name used in logs and result tables.
    name: str = "model"

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, warm_df: pd.DataFrame) -> "Model":
        """
        Train the model on warm-item interactions.

        Parameters
        ----------
        warm_df : DataFrame [user_id, item_id, engagement]
            Interactions of warm items only. All training hyperparameters are
            taken from the constructor, so this call stays model-agnostic.

        Returns self for chaining.
        """

    @abstractmethod
    def predict_batch(self, user_ids: list, item_ids: list) -> np.ndarray:
        """
        Predict scores for a list of (user_id, item_id) pairs.

        Parameters
        ----------
        user_ids : list of user IDs
        item_ids : list of item IDs (same length as user_ids)

        Returns
        -------
        scores : np.ndarray of shape (len(user_ids),), dtype float32.
                 Unknown IDs return 0.0.
        """

    # ------------------------------------------------------------------
    # Optional: embedding access (EmbeddingAggregator, LinMap, ...)
    # ------------------------------------------------------------------

    @property
    def supports_embeddings(self) -> bool:
        """
        True if this model exposes user/item embeddings.

        The runner checks this before running embedding-based methods and
        skips them (with a warning) on models that return False.
        """
        return False

    def get_item_embeddings_batch(self, item_ids: list) -> np.ndarray:
        """
        Embedding matrix for item_ids, shape (len(item_ids), dim).
        Override in models that support embeddings; default raises.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose item embeddings. "
            "Set supports_embeddings=True and implement this to use "
            "embedding-based cold-start methods."
        )

    def get_user_embeddings_batch(self, user_ids: list) -> np.ndarray:
        """
        Embedding matrix for user_ids, shape (len(user_ids), dim).
        Override in models that support embeddings; default raises.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose user embeddings. "
            "Set supports_embeddings=True and implement this to use "
            "embedding-based cold-start methods."
        )

    # ------------------------------------------------------------------
    # Optional: known-ID introspection (for honest validation)
    # ------------------------------------------------------------------

    @property
    def known_user_ids(self) -> list:
        """
        Users the model actually learned representations for.

        Used by the runner's generalisation check to evaluate AUC only on
        pairs the model can genuinely score. Empty list = "assume all known".
        """
        return []

    @property
    def known_item_ids(self) -> list:
        """
        Items the model actually learned representations for.
        Empty list = "assume all known".
        """
        return []
