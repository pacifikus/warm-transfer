# Installation

warm-transfer has a lightweight core and optional extras for benchmark engines and neural methods.
Python 3.11 or newer is required.

=== "pip"

    ```bash
    pip install warm-transfer            # core
    pip install "warm-transfer[bench]"   # benchmark donors and dataset tooling
    pip install "warm-transfer[all]"     # bench + deep extras
    ```

=== "uv"

    ```bash
    uv add warm-transfer            # core
    uv add "warm-transfer[bench]"   # benchmark donors and dataset tooling
    uv add "warm-transfer[all]"     # bench + deep extras
    ```

The package is published on [PyPI](https://pypi.org/project/warm-transfer/).

## Variants

| Variant | Install | Includes |
|---|---|---|
| Core | `warm-transfer` | `warmtransfer`: methods, metrics, similarity and public types |
| Benchmark | `warm-transfer[bench]` | donor engines, dataset downloads, parquet/YAML tooling and `warmbench` |
| Deep | `warm-transfer[deep]` | `torch` for neural cold-start methods such as `dropoutnet` |
| All | `warm-transfer[all]` | benchmark and deep dependencies |

## From source (development)

To work on warm-transfer itself, clone the repository and sync the environment:

```bash
uv sync                 # core + dev tools
uv sync --extra bench   # + benchmark donors and dataset tooling
uv sync --extra all     # + deep extras
```

## Smoke checks

```bash
uv run python examples/quickstart.py
uv run warmbench --list-components
```

The first command checks the core package. The second checks that the benchmark entrypoint and
registries are available.

!!! note
    `examples/quickstart.py` is only present in a repository clone; it is not shipped in the pip
    wheel. For a `pip` install, use the inline snippet on the [Quickstart](quickstart.md) page to
    smoke-check the core package instead.
