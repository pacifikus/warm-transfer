# Установка

У warm-transfer есть лёгкое ядро и optional extras для benchmark-движков и нейросетевых методов.
Нужен Python 3.11 или новее.

=== "uv"

    ```bash
    uv sync                 # core + dev tools
    uv sync --extra bench   # benchmark donors и dataset tooling
    uv sync --extra all     # bench + deep extras
    ```

=== "pip"

    ```bash
    python -m pip install warm-transfer
    python -m pip install "warm-transfer[bench]"
    python -m pip install "warm-transfer[all]"
    ```

## Варианты

| Вариант | Установка | Что входит |
|---|---|---|
| Core | `warm-transfer` | `warmtransfer`: методы, метрики, similarity и публичные типы |
| Benchmark | `warm-transfer[bench]` | donor engines, загрузка датасетов, parquet/YAML tooling и `warmbench` |
| Deep | `warm-transfer[deep]` | `torch` для нейросетевых cold-start методов вроде `dropoutnet` |
| All | `warm-transfer[all]` | benchmark и deep зависимости |

## Smoke checks

```bash
uv run python examples/quickstart.py
uv run warmbench --list-components
```

Первая команда проверяет core package. Вторая проверяет benchmark entrypoint и registries.
