"""CLI бенчмарка: ``coldbench --config configs/example.yaml``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coldbench", description="Прогон cold-start бенчмарка")
    parser.add_argument("--config", default=None, help="путь к YAML-конфигу прогона")
    parser.add_argument("--name", default="results", help="имя выходной таблицы")
    parser.add_argument(
        "--std", action="store_true", help="добавить колонки std по сидам"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="проверить конфиг и зарегистрированные компоненты без запуска бенчмарка",
    )
    parser.add_argument(
        "--list-components",
        action="store_true",
        help="показать зарегистрированные datasets/donors/methods/splitters",
    )
    parser.add_argument(
        "--rela-base",
        default=None,
        help="метод-бейзлайн для колонки RelaImpr (относительное улучшение AUC)",
    )
    args = parser.parse_args(argv)

    if args.list_components:
        _print_components()
        return 0
    if args.config is None:
        parser.error("--config обязателен, кроме режима --list-components")

    from coldbench.config import BenchConfig
    from coldbench.results import add_rela_impr, save_table, to_table
    from coldbench.runner import BenchmarkRunner

    config = BenchConfig.from_yaml(args.config)
    if args.dry_run:
        _validate_config_components(config)
        print("[coldbench] dry-run ok")
        print(f"datasets: {', '.join(config.datasets)}")
        print(f"donors: {', '.join(d.key for d in config.donors)}")
        print(f"methods: {', '.join(m.key for m in config.methods)}")
        print(f"splitter: {config.splitter.name}")
        return 0

    print(f"[coldbench] конфиг: {args.config}")
    records = BenchmarkRunner(config).run()
    table = to_table(records, with_std=args.std)
    if args.rela_base:
        table = add_rela_impr(table, base_method=args.rela_base)
    paths = save_table(table, config.out_dir, name=args.name)

    print("\n=== Результаты (усреднено по сидам) ===")
    print(table.to_markdown(index=False, floatfmt=".4f"))
    print(f"\nСохранено: {paths['markdown']}, {paths['parquet']}")
    return 0


def _print_components() -> None:
    from coldbench.adapters import adapters
    from coldbench.datasets import datasets
    from coldbench.splitters import splitters
    from coldscore import methods as _methods_pkg  # noqa: F401  # регистрирует методы
    from coldscore.methods import methods

    print("datasets:", ", ".join(datasets.names()))
    print("donors:", ", ".join(adapters.names()))
    print("methods:", ", ".join(methods.names()))
    print("splitters:", ", ".join(splitters.names()))


def _validate_config_components(config) -> None:
    from coldbench.adapters import adapters
    from coldbench.datasets import datasets
    from coldbench.splitters import splitters
    from coldscore import methods as _methods_pkg  # noqa: F401  # регистрирует методы
    from coldscore.methods import methods

    for name in config.datasets:
        datasets.get(name)
    for donor in config.donors:
        adapters.get(donor.name)
    for method in config.methods:
        methods.get(method.name)
    splitters.get(config.splitter.name)


if __name__ == "__main__":
    sys.exit(main())
