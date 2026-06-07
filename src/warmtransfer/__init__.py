"""warmtransfer — model-agnostic plug&play transfer/calibration of scores onto cold-start items.

Public API: unified column names, data types, the base method interface and registry,
metrics. No heavy dependencies are required — the library works with donor scores
provided by the user.
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
