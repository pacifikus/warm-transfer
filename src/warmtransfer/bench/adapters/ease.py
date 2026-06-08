"""Donor: EASE (Embarrassingly Shallow AutoEncoder, Steck WWW'2019).

A linear item-item model in CLOSED form — a family separate from matrix factorization
(als/bpr) and GBDT (catboost). Training is deterministic and almost hyperparameter-free:
the only one is the L2 regularization ``l2`` (λ).

Math:
    X  — binary interaction matrix [n_users, n_items] (implicit)
    G  = XᵀX + λ·I                       [n_items, n_items]
    P  = G⁻¹
    B  = -P / diag(P),  diag(B) := 0      (item→item weights, self-similarity zeroed)
    S  = X·B                              user scores over items

Pair score (u, i) = X[u] · B[:, i]. Cold items are absent from training (their columns
in B do not exist) — the donor scores only warm items, as the contract requires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from warmtransfer._pdutils import map_codes, unique_sorted
from warmtransfer.bench.adapters.base import ModelAdapter, register_adapter
from warmtransfer.columns import Columns as C
from warmtransfer.schema import validate_interactions
from warmtransfer.seeding import set_global_seed
from warmtransfer.types import Dataset


@register_adapter("ease")
class EASEAdapter(ModelAdapter):
    """EASE: a linear item-item autoencoder in closed form.

    :param l2: L2 regularization λ (diagonal addition). The only hyperparameter;
        a typical range is 100..1000. Larger λ → stronger smoothing.
    """

    def __init__(self, l2: float = 500.0) -> None:
        self.l2 = l2
        self._user_ids: np.ndarray | None = None
        self._item_ids: np.ndarray | None = None
        self._B: np.ndarray | None = None
        self._X: sparse.csr_matrix | None = None

    def fit(self, dataset: Dataset, seed: int = 0) -> EASEAdapter:
        set_global_seed(seed)
        inter = validate_interactions(dataset.interactions)

        self._user_ids = unique_sorted(inter[C.User])
        self._item_ids = unique_sorted(inter[C.Item])
        self._u_pos = {u: i for i, u in enumerate(self._user_ids)}
        self._i_pos = {it: j for j, it in enumerate(self._item_ids)}

        rows = map_codes(inter[C.User], self._u_pos)
        cols = map_codes(inter[C.Item], self._i_pos)
        n_u, n_i = len(self._user_ids), len(self._item_ids)
        # binary implicit matrix (canonical for EASE); duplicates collapse to 1.
        data = np.ones(len(rows), dtype=np.float64)
        X = sparse.csr_matrix((data, (rows, cols)), shape=(n_u, n_i))
        X.data[:] = 1.0
        self._X = X

        # G = XᵀX + λI  → P = G⁻¹  → B = -P / diag(P), diag(B)=0
        gram = np.asarray((X.T @ X).todense(), dtype=np.float64)
        gram[np.diag_indices(n_i)] += self.l2
        precision = np.linalg.inv(gram)
        diag = np.diag(precision).copy()
        b = -precision / diag  # divide each column j by diag[j]
        b[np.diag_indices(n_i)] = 0.0
        self._B = b
        return self

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> pd.DataFrame:
        """Scores for the cross product user_ids × item_ids (known warm only)."""
        if self._user_ids is None or self._item_ids is None or self._B is None or self._X is None:
            raise RuntimeError("EASEAdapter: call fit() before score()")

        u_known = [u for u in user_ids if u in self._u_pos]
        i_known = [it for it in item_ids if it in self._i_pos]
        u_idx = np.array([self._u_pos[u] for u in u_known], dtype=int)
        i_idx = np.array([self._i_pos[it] for it in i_known], dtype=int)

        # S = X[users] · B[:, items]   → [n_u, n_i]
        x_users = self._X[u_idx]  # sparse [n_u, n_items]
        b_items = self._B[:, i_idx]  # dense [n_items, n_i]
        scores = np.asarray(x_users @ b_items)  # dense [n_u, n_i]

        return pd.DataFrame(
            {
                C.User: np.repeat(u_known, len(i_known)),
                C.Item: np.tile(i_known, len(u_known)),
                C.Score: scores.reshape(-1),
            }
        )

    def get_params(self) -> dict:
        return {"l2": self.l2}
