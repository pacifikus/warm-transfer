# Установка

У warm-transfer есть лёгкое ядро и optional extras для benchmark-движков и нейросетевых методов.
Нужен Python 3.11 или новее.

=== "pip"

    ```bash
    pip install warm-transfer            # ядро
    pip install "warm-transfer[bench]"   # движки доноров и dataset tooling
    pip install "warm-transfer[all]"     # bench + deep extras
    ```

=== "uv"

    ```bash
    uv add warm-transfer            # ядро
    uv add "warm-transfer[bench]"   # движки доноров и dataset tooling
    uv add "warm-transfer[all]"     # bench + deep extras
    ```

Пакет опубликован на [PyPI](https://pypi.org/project/warm-transfer/).

## Варианты

| Вариант | Установка | Что входит |
|---|---|---|
| Core | `warm-transfer` | `warmtransfer`: методы, метрики, similarity и публичные типы |
| Benchmark | `warm-transfer[bench]` | donor engines, загрузка датасетов, parquet/YAML tooling и `warmbench` |
| Deep | `warm-transfer[deep]` | `torch` для нейросетевых cold-start методов вроде `dropoutnet` |
| All | `warm-transfer[all]` | benchmark и deep зависимости |

## Из исходников (для разработки)

Чтобы дорабатывать сам warm-transfer, склонируйте репозиторий и синхронизируйте окружение:

```bash
uv sync                 # ядро + dev-инструменты
uv sync --extra bench   # + движки доноров и инфраструктура бенчмарка
uv sync --extra all     # + deep extras
```

## Smoke checks

```bash
uv run python examples/quickstart.py
uv run warmbench --list-components
```

Первая команда проверяет core package. Вторая проверяет benchmark entrypoint и registries.

!!! note
    `examples/quickstart.py` есть только в клоне репозитория; в pip-wheel он не упаковывается. При
    установке через `pip` для smoke-check ядра используйте inline-сниппет со страницы
    [Быстрый старт](quickstart.md).
