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

## Main-command flags

The flat `warmbench` command (no subcommand) takes these flags:

| Flag | Default | Meaning |
|---|---|---|
| `--config` | — | path to the run's YAML config (required unless `--list-components`) |
| `--name` | `results` | output table name |
| `--std` | off | add per-seed std columns to the results table |
| `--rela-base` | — | baseline method for the `RelaImpr` column (relative AUC improvement) |
| `--dry-run` | off | validate the config and registered components without running |
| `--list-components` | off | print registered datasets/donors/methods/splitters |

Examples:

```bash
# add per-seed std columns and a custom table name
uv run warmbench --config configs/example.yaml --std --name als_run

# add a RelaImpr column relative to the Grouped MP baseline
uv run warmbench --config configs/example.yaml --rela-base grouped_most_popular_pers
```

## `warmbench try`: a quick verdict on your own data

`warmbench try` evaluates methods on data you provide and prints a leaderboard + verdict, without
the full dataset/donor matrix. It mirrors `warmtransfer.recommend()`. See the dedicated recipe in
[Quick verdict](quick-verdict.md) for the full walkthrough.

```bash
uv run warmbench try \
  --interactions inter.parquet \
  --content content.parquet \
  --scores scores.parquet
```

Required inputs (Parquet if the path ends in `.parquet`, otherwise CSV):

| Flag | Columns |
|---|---|
| `--interactions` | `user_id, item_id` (warm history) |
| `--content` | `item_id` + numeric feature columns |
| `--scores` | `user_id, item_id, score` (donor scores over warm items) |

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--metric` | `auc` | headline metric for ranking and verdict |
| `--methods` | `all` | comma-separated method names, or `all` |
| `--seeds` | `1` | number of seeds to average |
| `--seed` | `42` | base random seed |
| `--out` | — | write the leaderboard + verdict report to this file |

The command prints a ranked leaderboard by the chosen metric plus a one-line verdict on whether
transfer beats the baselines.

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
  - name: linmap
    label: linmap_a2          # same method, different hyperparameters, distinct results row
    params:
      alpha: 2.0

splitter:
  name: pseudo_cold
  params:
    cold_frac: 0.2
    val_frac: 0.1

metrics_ks: [1, 5, 10]
seeds: [42]
max_eval_users: 2000
out_dir: docs/results
```

Two extra keys are worth knowing:

- **`label`** (per component, optional) — name shown in the results table; defaults to `name`.
  Use it for ablations: list one method several times with different `params` under distinct
  labels so each appears as its own row in a single run.
- **`max_eval_users`** (top level, default `2000`) — caps how many users are scored during
  evaluation. Lower it for faster smoke runs; set it to `null` to evaluate all users.

## Read outputs

The runner writes Markdown/CSV/Parquet artifacts under `out_dir`. The main docs tables in
`docs/results/` are generated from those benchmark artifacts.

## Guardrails

- Run `--dry-run` after editing YAML.
- Keep `seeds` explicit.
- Use the same splitter and metric settings when comparing methods.
- Treat single-seed results as directional until multi-seed variance is checked.
