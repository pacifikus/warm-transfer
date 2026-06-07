"""Magnitude Scaling — post-hoc popularity debiasing based on embedding magnitude.

Changes only the length (norm) of the cold-item vector, not its direction. The embedding
magnitude is a proxy for popularity: items with large norms receive systematically high scores
(``score = user · item``). We pull the norm of each cold item toward the mean norm of warm items
``mu_w``, balancing the distribution of predictions across items:

    gamma_c = (||x_c|| + alpha * mu_w) / (||x_c|| * (1 + alpha)),
    x_c' = gamma_c * x_c    =>    ||x_c'|| = (||x_c|| + alpha * mu_w) / (1 + alpha),

that is, the new norm is a convex combination of ||x_c|| and mu_w with weight alpha/(1+alpha) on
mu_w (alpha→0: no change; alpha→inf: all norms pulled to mu_w).

The cold-embedding generator is taken from ``linmap_emb`` (Gantner): this gives a clean ablation
"linmap_emb vs +magnitude scaling" — the isolated effect of debiasing. It does not require
training the scaling itself, only statistics of the warm embeddings ([EMB]).

Reference: Meehan & Pauwels, "On Inherited Popularity Bias in Cold-Start Item
Recommendation", RecSys 2025.
"""

from __future__ import annotations

import numpy as np

from warmtransfer.methods.base import register_method
from warmtransfer.methods.linmap_emb import LinMapEmbedding


@register_method("magnitude_scaling")
class MagnitudeScaling(LinMapEmbedding):
    """Gantner generation of cold embeddings + popularity debiasing by magnitude.

    :param alpha: L2 regularization of Ridge (content → factors generator).
    :param ms_alpha: strength of pulling the norm toward ``mu_w`` (0 — no debiasing).
    """

    def __init__(self, alpha: float = 10.0, ms_alpha: float = 1.0) -> None:
        super().__init__(alpha=alpha)
        self.ms_alpha = ms_alpha

    def _postprocess_cold_emb(self) -> None:
        mu_w = self._warm_mean_norm
        norms = np.linalg.norm(self._cold_emb, axis=1)
        new_norms = (norms + self.ms_alpha * mu_w) / (1.0 + self.ms_alpha)
        # gamma = new_norm / norm; for a zero vector keep zero (no direction)
        gamma = np.divide(
            new_norms, norms, out=np.zeros_like(norms), where=norms > 0
        )
        self._cold_emb = self._cold_emb * gamma[:, None]

    def get_params(self) -> dict:
        return {"alpha": self.alpha, "ms_alpha": self.ms_alpha}
