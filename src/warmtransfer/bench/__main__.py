"""Benchmark CLI: ``warmbench --config configs/example.yaml``."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from warmtransfer.types import ItemFeatures

# Subcommands that use their own dedicated subparser
_SUBCOMMANDS = frozenset({"try"})


def main(argv: list[str] | None = None) -> int:
    # Detect if the first positional argument is a known subcommand and dispatch
    # to its dedicated parser; otherwise fall through to the existing flat parser
    # so that existing flags (--config, --list-components, …) keep working.
    args_list = list(argv) if argv is not None else sys.argv[1:]
    if args_list and args_list[0] in _SUBCOMMANDS:
        return _dispatch_subcommand(args_list)

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
    args = parser.parse_args(args_list)

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


def _dispatch_subcommand(args_list: list[str]) -> int:
    """Parse and dispatch a named subcommand."""
    parser = argparse.ArgumentParser(
        prog="warmbench", description="Run the cold-start benchmark"
    )
    sub = parser.add_subparsers(dest="command")

    p_try = sub.add_parser(
        "try",
        help="evaluate methods on your own data and print a verdict",
        description="Evaluate warm-transfer methods on your own data",
    )
    p_try.add_argument(
        "--interactions", required=True, help="interactions parquet/csv [user_id, item_id]"
    )
    p_try.add_argument(
        "--content",
        required=True,
        help="item content parquet/csv (item_id + numeric feature columns)",
    )
    p_try.add_argument(
        "--scores", required=True, help="donor scores parquet/csv [user_id, item_id, score]"
    )
    p_try.add_argument(
        "--metric", default="auc", help="headline metric for ranking and verdict (default: auc)"
    )
    p_try.add_argument(
        "--methods", default="all", help="comma-separated method names, or 'all' (default)"
    )
    p_try.add_argument(
        "--seeds", type=int, default=1, help="number of seeds to average (default: 1)"
    )
    p_try.add_argument("--seed", type=int, default=42, help="base random seed (default: 42)")
    p_try.add_argument(
        "--out", default=None, help="write the leaderboard+verdict report to this file"
    )
    p_try.set_defaults(func=_cmd_try)

    args = parser.parse_args(args_list)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)  # type: ignore[no-any-return]


def _load_table(path: str) -> pd.DataFrame:
    import pandas as pd

    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def _load_content(path: str) -> ItemFeatures:
    from warmtransfer.columns import Columns as C
    from warmtransfer.types import ItemFeatures

    df = _load_table(path)
    if C.Item not in df.columns:
        raise ValueError(f"Content file {path!r} is missing required column '{C.Item}'")
    feat_cols = [c for c in df.columns if c != C.Item]
    return ItemFeatures(
        item_ids=df[C.Item].to_numpy(),
        matrix=df[feat_cols].to_numpy(dtype=float),
        feature_names=feat_cols,
    )


def _cmd_try(args: argparse.Namespace) -> int:
    from warmtransfer import recommend
    from warmtransfer.holdout import HoldoutConfig

    interactions = _load_table(args.interactions)
    donor_scores = _load_table(args.scores)
    content = _load_content(args.content)
    methods = args.methods.split(",") if args.methods and args.methods != "all" else None
    result = recommend(
        interactions,
        content,
        donor_scores,
        metric=args.metric,
        methods=methods,
        holdout=HoldoutConfig(),
        seed=args.seed,
        n_seeds=args.seeds,
        verbose=False,
    )
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(str(result) + "\n")
    print(result)
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
