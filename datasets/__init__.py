"""Dataset registration (ML-1M, Amazon Toys, ...).

Importing this package registers all datasets with the core registry, so
get_dataset("ml1m") / get_dataset("amazon") work after `import datasets`.

Both datasets are already in the standard processed-CSV format, so they need
no bespoke loader class — one register() line each via the generic
ProcessedDataset. Genuine per-dataset differences live in the prep scripts
(datasets/prepare_*.py), not here.
"""

from warmtransfer.core.dataset import ProcessedDataset, register

register("ml1m", ProcessedDataset("MovieLens-1M", "data/processed"))
register("amazon", ProcessedDataset("Amazon Toys & Games", "data/processed/amazon_toys"))
