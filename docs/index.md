# warm-transfer

**Model-agnostic plug&play library** for transferring and calibrating the scores of an already
trained recommender model to **new (cold-start) items** under extreme sparsity, plus a
reproducible **benchmark**.

The idea: you have a trained model of arbitrary architecture — the library "wraps" around its
scores and rates new products/content for which there are no (or almost no) interactions yet.
You do not need to retrain the model, nor do you need access to its internals.

## Main result

Model-agnostic methods **LinMap** (Ridge: content → donor score vector) and **stacking_plus**
(hybrid: linmap signal + personalized popularity) **beat the strong personalized
Grouped MP** on the full matrix of **3 domains × 3 donors**:

- by **AUC** — in 7 of 9 cells (with the ALS donor — in all three domains);
- by **ranking** (NDCG@10, Recall@10) — on ML-1M and KION with all donors;
- the gaps exceed the spread across 5 seeds.

Naive methods (KNN averaging, attention, embedding-avg) lose to the baseline — they pull in
neighbors' popularity. Details and tables are in the [Results](results/full_matrix.md) section.

## Architecture

- **`warmtransfer`** — lightweight core (plug&play): transfer methods, metrics, similarity.
  Works with donor scores + content, installs without heavy recsys dependencies.
- **`warmtransfer.bench`** — benchmark (extra `bench`): datasets, fair splitter, donors
  (ALS/BPR/CatBoost), runner.

## Quick start (core)

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

## Quick verdict: which method fits my data?

Bring your interactions, item content and your model's scores — `recommend` evaluates every
feasible method on an honest pseudo-cold holdout and tells you the best one (and whether
transfer is worth it at all):

```python
import warmtransfer as wt

result = wt.recommend(interactions, content, donor_scores)
print(result)              # leaderboard + verdict
result.best_transfer       # best non-baseline method, e.g. "linmap"
result.predict(users, new_item_ids)   # winner, refit on all warm data
```

Or from the terminal:

```bash
warmbench try --interactions inter.parquet --content content.parquet --scores scores.parquet
```

The estimate uses a holdout of your warm items; the donor is not retrained, so treat it as a
mildly optimistic signal of whether transfer helps.

## Installation

```bash
uv sync                 # core + dev only
uv sync --extra bench   # + donor engines and benchmark
uv sync --extra all     # + deep (torch)
```

## Benchmark check

```bash
uv run python examples/quickstart.py
uv run warmbench --list-components
uv run warmbench --config configs/example.yaml --dry-run
uv run warmbench --config configs/example.yaml
```

See [Methods](methods.md), [Datasets](datasets.md), [Evaluation protocol](eval-protocol.md),
[API](api.md).
