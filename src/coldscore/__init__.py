"""coldscore — model-agnostic plug&play трансфер/калибровка скоров на cold-start айтемы.

Публичный API: единые имена колонок, типы данных, базовый интерфейс метода и реестр,
метрики. Тяжёлые зависимости не требуются — библиотека работает со скорами донора,
которые приносит пользователь.
"""

from __future__ import annotations

from coldscore.columns import Columns
from coldscore.methods.base import ColdStartMethod, methods, register_method
from coldscore.seeding import make_rng, set_global_seed
from coldscore.types import Dataset, ItemFeatures, SplitResult, TransferInputs

__all__ = [
    "ColdStartMethod",
    "Columns",
    "Dataset",
    "ItemFeatures",
    "SplitResult",
    "TransferInputs",
    "__version__",
    "make_rng",
    "methods",
    "register_method",
    "set_global_seed",
]
__version__ = "0.1.0"
