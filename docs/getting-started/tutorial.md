# Tutorial (MovieLens)

This tutorial follows the benchmark path on MovieLens-1M. It is a narrative version of the
`configs/example.yaml` run.

## 1. Choose the dataset

MovieLens-1M is small enough for quick iteration and has genre features for every movie.

```yaml
datasets:
  - ml-1m
```

## 2. Pick a donor

Start with ALS. It trains on warm interactions only after the pseudo-cold split.

```yaml
donors:
  - name: als
    params:
      factors: 64
      regularization: 0.05
      iterations: 20
```

## 3. Compare baselines and transfer

Grouped MP is the strong popularity baseline. `linmap` tests direct content-to-score transfer.
`knn_score_avg` is the naive neighbor baseline.

```yaml
methods:
  - name: grouped_most_popular_pers
  - name: linmap
  - name: knn_score_avg
    params:
      k: 20
```

## 4. Keep the split honest

```yaml
splitter:
  name: pseudo_cold
  params:
    cold_frac: 0.2
    val_frac: 0.1
    n_pop_buckets: 5
    min_item_interactions: 5
```

The splitter removes pseudo-cold items from donor training and neighbor construction. See the
[evaluation protocol](../eval-protocol.md) for the invariant.

## 5. Run

```bash
uv run warmbench --config configs/example.yaml --dry-run
uv run warmbench --config configs/example.yaml
```

Expected direction: `linmap` should be competitive with or stronger than the personalized Grouped MP
baseline on AUC, while naive KNN is a useful check for inherited popularity.

## 6. Read the metrics

Ranking metrics (`recall@k`, `precision@k`, `map@k`, `ndcg@k`, `mrr@k`) show top-k quality. AUC is
reported as an auxiliary per-user ranking signal. For final claims, compare across seeds rather than a
single run.
