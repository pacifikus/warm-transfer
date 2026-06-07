"""Feature-based donor: CatBoostClassifier on top of implicit feedback.

Trained on warm interactions as a binary classification task: positives are the
observed (user, item) pairs with label 1, negatives are randomly sampled
(user, random item) pairs with label 0. ``user_id`` and ``item_id`` are fed as
CATEGORICAL features. ``score`` returns ``predict_proba[:, 1]`` over the
cross-product of known (warm) users and items.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer._pdutils import unique_sorted
from warmtransfer.bench.adapters.base import ModelAdapter, register_adapter
from warmtransfer.columns import Columns as C
from warmtransfer.schema import validate_interactions
from warmtransfer.seeding import make_rng, set_global_seed
from warmtransfer.types import Dataset


@register_adapter("catboost")
class CatBoostAdapter(ModelAdapter):
    """CatBoostClassifier as a donor over categorical ids.

    :param iterations: number of CatBoost trees.
    :param depth: tree depth.
    :param neg_ratio: how many negatives to sample per positive.
    """

    def __init__(
        self,
        iterations: int = 100,
        depth: int = 6,
        neg_ratio: int = 4,
    ) -> None:
        self.iterations = iterations
        self.depth = depth
        self.neg_ratio = neg_ratio
        self._user_ids: np.ndarray | None = None
        self._item_ids: np.ndarray | None = None

    def fit(self, dataset: Dataset, seed: int = 0) -> CatBoostAdapter:
        from catboost import CatBoostClassifier, Pool

        set_global_seed(seed)
        rng = make_rng(seed)
        inter = validate_interactions(dataset.interactions)

        self._user_ids = unique_sorted(inter[C.User])
        self._item_ids = unique_sorted(inter[C.Item])
        self._u_set = set(self._user_ids.tolist())
        self._i_set = set(self._item_ids.tolist())

        pos_users = np.asarray(inter[C.User].to_numpy())
        pos_items = np.asarray(inter[C.Item].to_numpy())
        n_pos = len(pos_users)
        pos_pairs = set(zip(pos_users.tolist(), pos_items.tolist(), strict=True))

        # Negative sampling: same user, random item (excluding positives).
        n_neg = n_pos * self.neg_ratio
        neg_users = pos_users[rng.integers(0, n_pos, size=n_neg)]
        neg_items = self._item_ids[rng.integers(0, len(self._item_ids), size=n_neg)]
        neg_zip = zip(neg_users.tolist(), neg_items.tolist(), strict=True)
        keep = [pair not in pos_pairs for pair in neg_zip]
        neg_users = neg_users[keep]
        neg_items = neg_items[keep]

        users = np.concatenate([pos_users, neg_users])
        items = np.concatenate([pos_items, neg_items])
        labels = np.concatenate(
            [np.ones(n_pos, dtype=int), np.zeros(len(neg_users), dtype=int)]
        )

        # CatBoost requires categorical features as strings.
        x = pd.DataFrame(
            {
                C.User: users.astype(str),
                C.Item: items.astype(str),
            }
        )
        pool = Pool(x, label=labels, cat_features=[C.User, C.Item])

        self._model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            random_seed=seed,
            verbose=False,
        )
        self._model.fit(pool)
        return self

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> pd.DataFrame:
        """Scores for the cross-product user_ids x item_ids (known warm only)."""
        if self._user_ids is None or self._item_ids is None:
            raise RuntimeError("CatBoostAdapter: call fit() before score()")

        u_known = [u for u in user_ids if u in self._u_set]
        i_known = [it for it in item_ids if it in self._i_set]
        if not u_known or not i_known:
            return pd.DataFrame({C.User: [], C.Item: [], C.Score: []})

        users = np.repeat(u_known, len(i_known))
        items = np.tile(i_known, len(u_known))
        x = pd.DataFrame({C.User: users.astype(str), C.Item: items.astype(str)})
        proba = self._model.predict_proba(x)[:, 1]

        return pd.DataFrame(
            {
                C.User: users,
                C.Item: items,
                C.Score: np.asarray(proba, dtype=float),
            }
        )

    def embeddings(self) -> dict[str, np.ndarray] | None:
        return None

    def get_params(self) -> dict:
        return {
            "iterations": self.iterations,
            "depth": self.depth,
            "neg_ratio": self.neg_ratio,
        }
