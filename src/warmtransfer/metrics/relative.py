"""RelaImpr — relative AUC improvement over a baseline.

RelaImpr = (AUC_model − 0.5) / (AUC_base − 0.5) − 1, often ×100 for percentages.
Measures the improvement "over random" relative to the baseline (a CTR/recsys standard).
"""

from __future__ import annotations

import math


def rela_impr(model_auc: float, base_auc: float) -> float:
    """Relative AUC improvement of the model over the baseline (a fraction, not percent)."""
    denom = base_auc - 0.5
    if denom <= 0:
        return math.nan
    return (model_auc - 0.5) / denom - 1.0
