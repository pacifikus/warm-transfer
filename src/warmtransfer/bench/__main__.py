"""Benchmark CLI: ``warmbench --config configs/example.yaml``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="warmbench", description="Run the cold-start benchmark")
    parser.add_argument("--config", default=None, help="path to the run's YAML config")
    parser.add_argument("--name", default="results", help="output table name")
    parser.add_argument(
        "--std", action="store_true", help="add per-seed std columns"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the config and registered components without running the benchmark",
    )
    parser.add_argument(
        "--list-components",
        action="store_true",
        help="show registered datasets/donors/methods/splitters",
    )
    parser.add_argument(
        "--rela-base",
        default=None,
        help="baseline method for the RelaImpr column (relative AUC improvement)",
    )
    args = parser.parse_args(argv)

    if args.list_components:
        _print_components()
        return 0
    if args.config is None:
        parser.error("--config is required except in --list-components mode")

    from warmtransfer.bench.config import BenchConfig
    from warmtransfer.bench.results import add_rela_impr, save_table, to_table
    from warmtransfer.bench.runner import BenchmarkRunner

    config = BenchConfig.from_yaml(args.config)
    if args.dry_run:
        _validate_config_components(config)
        print("[warmbench] dry-run ok")
        print(f"datasets: {', '.join(config.datasets)}")
        print(f"donors: {', '.join(d.key for d in config.donors)}")
        print(f"methods: {', '.join(m.key for m in config.methods)}")
        print(f"splitter: {config.splitter.name}")
        return 0

    print(f"[warmbench] config: {args.config}")
    records = BenchmarkRunner(config).run()
    table = to_table(records, with_std=args.std)
    if args.rela_base:
        table = add_rela_impr(table, base_method=args.rela_base)
    paths = save_table(table, config.out_dir, name=args.name)

    print("\n=== Results (averaged over seeds) ===")
    print(table.to_markdown(index=False, floatfmt=".4f"))
    print(f"\nSaved: {paths['markdown']}, {paths['parquet']}")
    return 0


def _print_components() -> None:
    from warmtransfer import methods as _methods_pkg  # noqa: F401  # registers methods
    from warmtransfer.bench.adapters import adapters
    from warmtransfer.bench.datasets import datasets
    from warmtransfer.bench.splitters import splitters
    from warmtransfer.methods import methods

    print("datasets:", ", ".join(datasets.names()))
    print("donors:", ", ".join(adapters.names()))
    print("methods:", ", ".join(methods.names()))
    print("splitters:", ", ".join(splitters.names()))


def _validate_config_components(config) -> None:
    from warmtransfer import methods as _methods_pkg  # noqa: F401  # registers methods
    from warmtransfer.bench.adapters import adapters
    from warmtransfer.bench.datasets import datasets
    from warmtransfer.bench.splitters import splitters
    from warmtransfer.methods import methods

    for name in config.datasets:
        datasets.get(name)
    for donor in config.donors:
        adapters.get(donor.name)
    for method in config.methods:
        methods.get(method.name)
    splitters.get(config.splitter.name)


if __name__ == "__main__":
    sys.exit(main())
