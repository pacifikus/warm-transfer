"""warmtransfer — model-agnostic plug&play трансфер/калибровка скоров на cold-start айтемы.

Публичный API: единые имена колонок, типы данных, базовый интерфейс метода и реестр,
метрики. Тяжёлые зависимости не требуются — библиотека работает со скорами донора,
которые приносит пользователь.
"""

from __future__ import annotations

from warmtransfer.columns import Columns
from warmtransfer.methods.base import ColdStartMethod, methods, register_method
from warmtransfer.seeding import make_rng, set_global_seed
from warmtransfer.types import Dataset, ItemFeatures, SplitResult, TransferInputs

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
