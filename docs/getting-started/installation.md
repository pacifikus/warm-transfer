# Installation

warm-transfer has a lightweight core and optional extras for benchmark engines and neural methods.
Python 3.11 or newer is required.

=== "uv"

    ```bash
    uv sync                 # core + dev tools
    uv sync --extra bench   # benchmark donors and dataset tooling
    uv sync --extra all     # bench + deep extras
    ```

=== "pip"

    ```bash
    python -m pip install warm-transfer
    python -m pip install "warm-transfer[bench]"
    python -m pip install "warm-transfer[all]"
    ```

## Variants

| Variant | Install | Includes |
|---|---|---|
| Core | `warm-transfer` | `warmtransfer`: methods, metrics, similarity and public types |
| Benchmark | `warm-transfer[bench]` | donor engines, dataset downloads, parquet/YAML tooling and `warmbench` |
| Deep | `warm-transfer[deep]` | `torch` for neural cold-start methods such as `dropoutnet` |
| All | `warm-transfer[all]` | benchmark and deep dependencies |

## Smoke checks

```bash
uv run python examples/quickstart.py
uv run warmbench --list-components
```

The first command checks the core package. The second checks that the benchmark entrypoint and
registries are available.
