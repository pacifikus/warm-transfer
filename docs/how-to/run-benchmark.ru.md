# Как запустить бенчмарк

`warmbench` запускает datasets × donors × methods под единым pseudo-cold protocol.

## Посмотреть доступные компоненты

```bash
uv run warmbench --list-components
```

## Начать с example config

```bash
uv run warmbench --config configs/example.yaml --dry-run
uv run warmbench --config configs/example.yaml
```

Example запускает ML-1M × ALS на baselines и `knn_score_avg`.

## Форма config

```yaml
datasets:
  - ml-1m

donors:
  - name: als
    params:
      factors: 64

methods:
  - name: grouped_most_popular_pers
  - name: linmap

splitter:
  name: pseudo_cold
  params:
    cold_frac: 0.2
    val_frac: 0.1

metrics_ks: [1, 5, 10]
seeds: [42]
out_dir: docs/results
```

## Читать outputs

Runner пишет Markdown/CSV/Parquet artifacts в `out_dir`. Основные таблицы в `docs/results/`
собираются из этих benchmark artifacts.

## Guardrails

- Запускайте `--dry-run` после изменения YAML.
- Держите `seeds` явными.
- Используйте одинаковые splitter и metric settings при сравнении методов.
- Считайте single-seed results directional, пока не проверен multi-seed variance.
