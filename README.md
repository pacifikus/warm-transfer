**English** | [Русский](README.ru.md)

# warm-transfer

**Model-agnostic plug&play library** for transferring and calibrating the scores of an
already-trained recommendation model onto **new (cold-start) items** under extreme
sparsity, plus a reproducible **benchmark**.

Idea: you have a trained model of arbitrary architecture — the library "wraps" on top
of its scores and ranks new products/content for which there are still no (or almost no)
interactions. You don't need to retrain the model, and you don't need access to its
internals either.

## Structure

- **`warmtransfer`** — lightweight core (plug&play). Works with donor scores + content.
  Installs without heavy recsys dependencies.
  - `methods/` — cold-start methods (baselines, KNN, LinMap, Stacking, scale&shift, attention-KNN…)
  - `metrics/` — our own correct metrics (Recall/Precision/MAP/NDCG@k, MRR, AUC, RelaImpr)
  - `similarity/` — content similarity cold→warm (optional)
- **`warmtransfer.bench`** — benchmark (heavy dependencies, extra `bench`).
  - `datasets/` — loaders (ML-1M/20M, Goodbooks, KION, KION-text)
  - `splitters/` — honest pseudo-cold split (anti-leakage)
  - `adapters/` — donor models (ALS, BPR, CatBoost)
  - `runner.py` — running the matrix of datasets × donors × methods × baselines

## Key result

The model-agnostic methods **LinMap** (Ridge: content → donor score vector) and **stacking_plus**
(hybrid: linmap signal + personalized popularity) **beat the strong personalized
Grouped MP** on the full matrix of **3 domains × 3 donors**: on AUC in 7 of 9 cells, and on ranking
on ML-1M and KION with all donors. The gaps exceed the spread across 5 seeds. Naive methods
(knn/attention/debiased/embedding_avg) lose to the baseline — they pull in neighbor popularity.
DropoutNet (deep [EMB]) gives the best ranking on the dense ML-1M. Details and tables —
`docs/results/full_matrix.md`.

## Plug&play usage (core, without the benchmark)

```python
import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.methods import LinMap
from warmtransfer.types import ItemFeatures, TransferInputs

warm_features = ItemFeatures(
    item_ids=np.array([10, 11]),
    matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
    feature_names=["genre_action", "genre_drama"],
)
cold_features = ItemFeatures(
    item_ids=np.array([20]),
    matrix=np.array([[1.0, 0.0]]),
    feature_names=["genre_action", "genre_drama"],
)
donor_scores = pd.DataFrame(
    {
        C.User: [1, 1, 2, 2],
        C.Item: [10, 11, 10, 11],
        C.Score: [5.0, 1.0, 1.0, 5.0],
    }
)

# bring your own warm donor scores + content of warm/cold items
inputs = TransferInputs(
    donor_scores=donor_scores,
    warm_features=warm_features,
    cold_features=cold_features,
)
reco = LinMap(alpha=1.0).fit(inputs, seed=42).predict(
    user_ids=np.array([1, 2]),
    cold_item_ids=np.array([20]),
)
```

Full runnable example: `examples/quickstart.py`.

## Installation

```bash
uv sync                 # core + dev only
uv sync --extra bench   # + donor engines and benchmark infrastructure
uv sync --extra all     # + deep (torch)
```

## Running

```bash
uv run pytest -q                          # core tests
uv run python examples/quickstart.py      # minimal plug&play example
uv run warmbench --list-components        # available datasets/donors/methods
uv run warmbench --config configs/example.yaml --dry-run
uv run warmbench --config configs/example.yaml  # example benchmark run
```

## Documentation

The documentation site is built with MkDocs Material (like RecTools) and published to
GitHub Pages automatically (`.github/workflows/docs.yml`) after a push to the GitHub repository.
Locally:

```bash
uv sync --group docs
uv run mkdocs serve     # local preview at http://127.0.0.1:8000
uv run mkdocs build     # static site in site/
```

Page sources:
- `docs/methods.md` — description of the methods
- `docs/datasets.md` — description of the datasets
- `docs/eval-protocol.md` — split and metrics protocol (anti-leakage)
- `docs/results/` — result tables (full matrix, spread across seeds, ablations)
- `docs/api.md` — auto-generated API reference
