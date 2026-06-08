# Быстрый старт

Эта страница показывает core plug&play путь: принести скоры донора по warm items, обучить один
метод переноса и предсказать скоры для cold-start items. `warmtransfer.bench` здесь не нужен.

## Установка

=== "uv"

    ```bash
    uv sync
    ```

=== "pip"

    ```bash
    python -m pip install warm-transfer
    ```

## Запустите пример

Полный исполняемый скрипт лежит в репозитории и подключается в документацию напрямую, чтобы пример в
доке не расходился с кодом.

```python
--8<-- "examples/quickstart.py"
```

Ожидаемый вывод:

```text
 user_id  item_id  score
       1       20    4.0
       2       20    2.0
```

## Что произошло

1. `donor_scores` — long-format таблица `[user_id, item_id, score]` только по warm items.
2. `warm_features` и `cold_features` связывают item ids с content-векторами.
3. `LinMap.fit(inputs, seed=42)` учит линейное отображение из контента айтема в вектор скоров донора.
4. `predict(user_ids, cold_item_ids)` возвращает long-format скоры по всем заданным парам user-item.

## Дальше

- Нужны варианты установки? Читайте [Установка](installation.md).
- Выбираете другой метод? Используйте [матрицу возможностей](../methods.md).
- Добавляете свой метод? Следуйте [Как добавить свой метод](../how-to/add-method.md).
- Запускаете бенчмарк? Следуйте [Как запустить бенчмарк](../how-to/run-benchmark.md).
