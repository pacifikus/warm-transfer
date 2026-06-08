# Quick verdict: which method fits my data?

This is the main onboarding workflow. You bring three things and `recommend()` tells you which
cold-start method actually works on *your* data, then scores new items for you.

You need:

- **interactions** — warm user history, long-format `[user_id, item_id]`.
- **content** — item features for all items (an `ItemFeatures` with `item_ids` + a numeric matrix).
- **donor scores** — your trained model's scores over warm items, `[user_id, item_id, score]`.

`recommend()` evaluates every feasible method on an honest pseudo-cold holdout of your warm items
and returns a leaderboard plus a verdict (is transfer worth it at all?). The donor is **not**
retrained, so treat the estimate as a mildly optimistic signal.

## From Python

```python
import warmtransfer as wt

result = wt.recommend(interactions, content, donor_scores, seed=42)

print(result)              # leaderboard + verdict
print(result.best)         # absolute best method on the holdout
print(result.best_transfer)  # best non-baseline method, e.g. "linmap"
```

Then score new cold items with the winner, refit on all your warm data:

```python
import numpy as np

reco = result.predict(user_ids=np.array([0, 1, 2]), cold_item_ids=np.array([0]))
print(reco.to_string(index=False))
```

`result.predict(...)` uses `result.best_transfer` (the best non-baseline method), refit on the
full warm dataset — the holdout was only for ranking.

A complete runnable version lives in `examples/recommend_quickstart.py`.

## From the terminal

The CLI mirrors the Python call. It loads three files, runs the same evaluation and prints the
verdict + leaderboard:

```bash
uv run warmbench try \
  --interactions inter.parquet \
  --content content.parquet \
  --scores scores.parquet
```

### Input files

| Flag | Required | Columns | Notes |
|---|---|---|---|
| `--interactions` | yes | `user_id, item_id` | warm history |
| `--content` | yes | `item_id` + numeric feature columns | every other column is read as a float feature |
| `--scores` | yes | `user_id, item_id, score` | donor scores over warm items |

Each file is read as Parquet if its path ends in `.parquet`, otherwise as CSV.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--metric` | `auc` | headline metric used to rank methods and form the verdict |
| `--methods` | `all` | comma-separated method names, or `all` |
| `--seeds` | `1` | number of seeds to average |
| `--seed` | `42` | base random seed |
| `--out` | — | also write the leaderboard + verdict report to this file |

Example with a restricted method set, three seeds and a saved report:

```bash
uv run warmbench try \
  --interactions inter.csv \
  --content content.csv \
  --scores scores.csv \
  --methods linmap,stacking_plus,grouped_most_popular_pers \
  --metric auc \
  --seeds 3 \
  --out verdict.txt
```

The command prints the same object as the Python `print(result)`: a ranked leaderboard of methods
by the chosen metric and a one-line verdict on whether transfer beats the baselines.

## Where to go next

- See how the methods differ: [Choose a method](choose-method.md).
- Run the full reproducible matrix: [Run the benchmark](run-benchmark.md).
