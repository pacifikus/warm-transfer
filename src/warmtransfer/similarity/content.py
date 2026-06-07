"""Content similarity of cold->warm items (cosine).

Can be delegated to the library user (they pass a ready-made similarity matrix),
or built here from ``ItemFeatures``.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from warmtransfer.types import ItemFeatures


def content_similarity(cold: ItemFeatures, warm: ItemFeatures) -> np.ndarray:
    """Cosine similarity [n_cold, n_warm] between cold and warm items.

    Rows are aligned to ``cold.item_ids``, columns to ``warm.item_ids``.
    """
    cold_mat = np.asarray(cold.matrix, dtype=float)
    warm_mat = np.asarray(warm.matrix, dtype=float)
    return cosine_similarity(cold_mat, warm_mat)
