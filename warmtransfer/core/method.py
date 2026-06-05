"""
ColdStartMethod — abstract base for all cold-start predictors and baselines,
plus the Context object that carries the shared "warm world" to every method.

Why Context exists
------------------
Before this refactor, every method had a different signature: some needed
warm_df, some item_features, some the model, some a score_fn. The runner had
to call each one by hand. Context fixes that: the runner builds it once and
hands the same object to every method, so all methods share one interface:

    method.fit(context)          # optional training (default: just store context)
    method.predict(cold_df)      # -> standard predictions DataFrame

A method pulls only what it needs out of the context. Adding a new method is
then a single self-contained class — no edits to the runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:                      # avoid import cycles at runtime
    from warmtransfer.core.model import Model
    from warmtransfer.similarity.item_similarity import ItemSimilarity


# ---------------------------------------------------------------------------
# Context — the shared "warm world" handed to every method
# ---------------------------------------------------------------------------

@dataclass
class Context:
    """
    Everything a cold-start method might need, assembled once by the runner.

    Fields
    ------
    warm_df         : interactions of warm items   [user_id, item_id, engagement]
    pseudo_cold_df  : warm interactions held out to TRAIN calibrator-like methods
                      (the model has seen these items; we simulate them as cold)
    warm_item_ids   : full pool of warm item IDs (neighbour candidates)
    item_features   : DataFrame [item_id, <feature columns>] for ALL items
    model           : the trained Model (black box)
    similarity      : content-similarity index over item_features
    """

    warm_df: pd.DataFrame
    pseudo_cold_df: pd.DataFrame
    warm_item_ids: list
    item_features: pd.DataFrame
    model: "Model"
    similarity: "ItemSimilarity"

    @property
    def score_fn(self) -> Callable[[list, list], np.ndarray]:
        """Convenience: the model's batch scoring function."""
        return self.model.predict_batch


# ---------------------------------------------------------------------------
# ColdStartMethod — the uniform interface
# ---------------------------------------------------------------------------

class ColdStartMethod(ABC):
    """
    Abstract cold-start predictor.

    Lifecycle (driven by the runner)
    --------------------------------
        method.fit(context)        # store context; train if needed
        preds = method.predict(cold_df)

    Subclasses MUST implement predict(). They override fit() only if they need
    to train (e.g. the score-aggregation calibrator); the default fit() just
    stores the context, which is enough for stateless methods and baselines.

    Class attributes
    ----------------
    name                : label shown in result tables
    requires_embeddings : if True, the runner skips this method on models that
                          do not expose embeddings (model.supports_embeddings)
    """

    name: str = "unnamed-method"
    requires_embeddings: bool = False

    def __init__(self) -> None:
        self._context: Context | None = None

    # ------------------------------------------------------------------

    def fit(self, context: Context) -> "ColdStartMethod":
        """
        Prepare the method against the warm world.

        Default implementation just stores the context. Methods that train
        (calibrators) override this, call super().fit(context), then train.
        """
        self._context = context
        return self

    @abstractmethod
    def predict(self, cold_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score every (user, item) pair in cold_df.

        Parameters
        ----------
        cold_df : DataFrame [user_id, item_id, engagement]

        Returns
        -------
        DataFrame [user_id, item_id, engagement, predicted_score]
        """

    # ------------------------------------------------------------------

    @property
    def context(self) -> Context:
        """The context stored by fit(). Raises if fit() was not called."""
        if self._context is None:
            raise RuntimeError(
                f"{type(self).__name__}.predict() called before fit(). "
                "The runner calls fit(context) first."
            )
        return self._context
