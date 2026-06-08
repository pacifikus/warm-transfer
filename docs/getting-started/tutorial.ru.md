# Туториал (MovieLens)

Этот туториал проходит benchmark path на MovieLens-1M. Это narrative-версия запуска
`configs/example.yaml`.

## 1. Выбрать dataset

MovieLens-1M достаточно маленький для быстрых итераций и содержит genre features для каждого фильма.

```yaml
datasets:
  - ml-1m
```

## 2. Выбрать donor

Начните с ALS. Он обучается только на warm interactions после pseudo-cold split.

```yaml
donors:
  - name: als
    params:
      factors: 64
      regularization: 0.05
      iterations: 20
```

## 3. Сравнить baselines и transfer

Grouped MP — сильный popularity baseline. `linmap` проверяет direct content-to-score transfer.
`knn_score_avg` — naive neighbor baseline.

```yaml
methods:
  - name: grouped_most_popular_pers
  - name: linmap
  - name: knn_score_avg
    params:
      k: 20
```

## 4. Сохранить честный split

```yaml
splitter:
  name: pseudo_cold
  params:
    cold_frac: 0.2
    val_frac: 0.1
    n_pop_buckets: 5
    min_item_interactions: 5
```

Splitter удаляет pseudo-cold items из donor training и neighbor construction. См.
[evaluation protocol](../eval-protocol.md) для invariant.

## 5. Запустить

```bash
uv run warmbench --config configs/example.yaml --dry-run
uv run warmbench --config configs/example.yaml
```

Ожидаемое направление: `linmap` должен быть competitive with or stronger than personalized Grouped MP
baseline по AUC, а naive KNN полезен как проверка inherited popularity.

## 6. Читать metrics

Ranking metrics (`recall@k`, `precision@k`, `map@k`, `ndcg@k`, `mrr@k`) показывают top-k quality.
AUC репортится как auxiliary per-user ranking signal. Для финальных claims сравнивайте результаты по
seeds, а не по одному run.
