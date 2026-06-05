"""Core abstractions: Dataset, Model, ColdStartMethod, Context, BenchmarkRunner.

Note: BenchmarkRunner is intentionally NOT re-exported here. It imports the
splitters/similarity/metrics subpackages; importing it before those modules
exist would break `import warmtransfer.core`. Import it directly when needed:

    from warmtransfer.core.runner import BenchmarkRunner
"""

from warmtransfer.core.model import Model
from warmtransfer.core.dataset import (
    Dataset,
    ProcessedDataset,
    register,
    register_dataset,
    get_dataset,
    list_datasets,
)
from warmtransfer.core.method import ColdStartMethod, Context
from warmtransfer.core.runner import BenchmarkRunner

__all__ = [
    "Model",
    "Dataset",
    "ProcessedDataset",
    "register",
    "register_dataset",
    "get_dataset",
    "list_datasets",
    "ColdStartMethod",
    "Context",
    "BenchmarkRunner",
]
