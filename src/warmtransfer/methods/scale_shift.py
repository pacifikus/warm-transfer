"""Scale & Shift — a lightweight, score-only flavor of MWUF.

MWUF (Zhu et al., SIGIR 2021) maps a cold ID embedding into the "warm" space with two
networks: scaling and shifting. Here we transfer that idea onto SCORES, without access to
embeddings and without training any networks.

The base raw score of a cold item is built like in naive KNN — the mean of the donor scores
over the k content-based warm neighbors. Then, for each cold item, we standardize its score
vector over users and rescale it to the typical warm statistics (scale ``sigma*``, shift
``mu*``):

    raw[u, c]  = Σ_j w_{cj} · donor[u, j]        (content-based KNN)
    out[u, c]  = (raw[u, c] - m_c) / (s_c + eps) · sigma* + mu*,

where ``m_c, s_c`` are the mean/std of the cold item's raw score over users, and ``mu*,
sigma*`` are the warm-item averages of the per-item mean/std of their scores. Effect: the
inflated overall score level of popular cold items is removed (debiasing), and the
distribution is fitted to the warm one. Only donor scores + content (for neighbors) are
needed, no embeddings ([MA], score-only).

Reference (the scale/shift idea): Zhu et al., "Learning to Warm Up Cold Item Embeddings for
Cold-start Recommendation with Meta Scaling and Shifting Networks", SIGIR 2021.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.methods.base import ColdStartMethod, cross_join_frame, register_method
from warmtransfer.methods.debiased_knn import _pivot_scores, _topk_neighbors
from warmtransfer.types import TransferInputs

_EPS = 1e-8


@register_method("scale_shift")
class ScaleShift(ColdStartMethod):
    """Content-based KNN over donor scores + calibration to warm statistics.

    :param k: number of nearest warm neighbors.
    :param clip_negative: zero out negative similarities.
    """

    requires = frozenset({"donor_scores", "similarity", "content"})

    def __init__(self, k: int = 20, clip_negative: bool = True) -> None:
        super().__init__()
        self.k = k
        self.clip_negative = clip_negative

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        if inputs.warm_features is None or inputs.cold_features is None:
            raise ValueError("scale_shift requires warm_features and cold_features")
        if inputs.similarity is None:
            raise ValueError("scale_shift requires similarity [n_cold, n_warm]")

        self._warm_ids = np.asarray(inputs.warm_features.item_ids)
        self._cold_ids = np.asarray(inputs.cold_features.item_ids)
        self._cold_pos = {it: r for r, it in enumerate(self._cold_ids)}

        # donor score matrix [n_users, n_warm], aligned with self._warm_ids
        donor, self._user_ids = _pivot_scores(inputs.donor_scores, self._warm_ids)
        self._user_pos = {u: i for i, u in enumerate(self._user_ids)}

        # target warm statistics: item-wise averages of the per-user mean/std of the score
        self._mu_star = float(donor.mean(axis=0).mean())
        self._sigma_star = float(donor.std(axis=0).mean())

        sim = np.asarray(inputs.similarity, dtype=float)
        if self.clip_negative:
            sim = np.clip(sim, 0.0, None)
        neighbors, weights = _topk_neighbors(sim, self.k)

        # raw cold-item score [n_users, n_cold] = KNN averaging of donor scores
        raw = np.zeros((donor.shape[0], len(self._cold_ids)))
        for r in range(len(self._cold_ids)):
            raw[:, r] = donor[:, neighbors[r]] @ weights[r]

        # standardize over users + scale&shift to the warm statistics
        m_c = raw.mean(axis=0)  # [n_cold]
        s_c = raw.std(axis=0)  # [n_cold]
        z = (raw - m_c) / (s_c + _EPS)
        self._cold_scores = z * self._sigma_star + self._mu_star  # [n_users, n_cold]

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        user_ids = np.asarray(user_ids)
        cold_item_ids = np.asarray(cold_item_ids)

        rows = np.array([self._user_pos.get(u, -1) for u in user_ids])
        cols = np.array([self._cold_pos.get(it, -1) for it in cold_item_ids])

        scores = np.zeros((len(user_ids), len(cold_item_ids)))
        u_known = rows >= 0
        for c, col in enumerate(cols):
            if col < 0:
                continue
            # unknown user → mu* (neutral warm level)
            scores[~u_known, c] = self._mu_star
            scores[u_known, c] = self._cold_scores[rows[u_known], col]
        return cross_join_frame(user_ids, cold_item_ids, scores)

    def get_params(self) -> dict:
        return {"k": self.k, "clip_negative": self.clip_negative}
