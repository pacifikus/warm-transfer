"""RelaImpr — относительное улучшение AUC над бейзлайном.

RelaImpr = (AUC_model − 0.5) / (AUC_base − 0.5) − 1, в процентах часто ×100.
Меряет улучшение «над случайным» относительно бейзлайна (стандарт в CTR/recsys).
"""

from __future__ import annotations

import math


def rela_impr(model_auc: float, base_auc: float) -> float:
    """Относительное улучшение AUC модели над бейзлайном (доля, не проценты)."""
    denom = base_auc - 0.5
    if denom <= 0:
        return math.nan
    return (model_auc - 0.5) / denom - 1.0
