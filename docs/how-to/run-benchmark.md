# Run the benchmark

`warmbench` runs datasets × donors × methods under the same pseudo-cold protocol.

## Inspect available components

```bash
uv run warmbench --list-components
```

## Start from the example config

```bash
uv run warmbench --config configs/example.yaml --dry-run
uv run warmbench --config configs/example.yaml
```

The example runs ML-1M × ALS on baselines and `knn_score_avg`.

## Config shape

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

## Read outputs

The runner writes Markdown/CSV/Parquet artifacts under `out_dir`. The main docs tables in
`docs/results/` are generated from those benchmark artifacts.

## Guardrails

- Run `--dry-run` after editing YAML.
- Keep `seeds` explicit.
- Use the same splitter and metric settings when comparing methods.
- Treat single-seed results as directional until multi-seed variance is checked.
