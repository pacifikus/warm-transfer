# Быстрый вердикт: какой метод подходит под мои данные?

Это главный onboarding-workflow. Вы приносите три вещи, а `recommend()` говорит, какой
cold-start метод реально работает на *ваших* данных, и сразу скорит новые айтемы.

Что нужно:

- **interactions** — тёплая история пользователей, long-format `[user_id, item_id]`.
- **content** — признаки всех айтемов (`ItemFeatures` с `item_ids` + числовая матрица).
- **donor scores** — скоры вашей обученной модели по тёплым айтемам, `[user_id, item_id, score]`.

`recommend()` оценивает каждый применимый метод на честном pseudo-cold holdout из ваших тёплых
айтемов и возвращает leaderboard плюс вердикт (стоит ли вообще трансфер?). Донор при этом **не**
переобучается, поэтому считайте оценку слегка оптимистичным сигналом.

## Из Python

```python
import warmtransfer as wt

result = wt.recommend(interactions, content, donor_scores, seed=42)

print(result)              # leaderboard + вердикт
print(result.best)         # абсолютно лучший метод на holdout
print(result.best_transfer)  # лучший не-baseline метод, например "linmap"
```

Затем скорим новые cold-айтемы победителем, переобучив его на всех тёплых данных:

```python
import numpy as np

reco = result.predict(user_ids=np.array([0, 1, 2]), cold_item_ids=np.array([0]))
print(reco.to_string(index=False))
```

`result.predict(...)` использует `result.best_transfer` (лучший не-baseline метод), переобученный
на полном тёплом датасете — holdout нужен был только для ранжирования.

Полный запускаемый пример лежит в `examples/recommend_quickstart.py`.

## Из терминала

CLI повторяет вызов из Python. Он загружает три файла, запускает ту же оценку и печатает
вердикт + leaderboard:

```bash
uv run warmbench try \
  --interactions inter.parquet \
  --content content.parquet \
  --scores scores.parquet
```

### Входные файлы

| Флаг | Обязателен | Колонки | Заметки |
|---|---|---|---|
| `--interactions` | да | `user_id, item_id` | тёплая история |
| `--content` | да | `item_id` + числовые колонки признаков | каждая остальная колонка читается как float-признак |
| `--scores` | да | `user_id, item_id, score` | скоры донора по тёплым айтемам |

Файл читается как Parquet, если путь оканчивается на `.parquet`, иначе как CSV.

### Опции

| Флаг | Дефолт | Значение |
|---|---|---|
| `--metric` | `auc` | основная метрика для ранжирования методов и вердикта |
| `--methods` | `all` | имена методов через запятую или `all` |
| `--seeds` | `1` | сколько сидов усреднять |
| `--seed` | `42` | базовый random seed |
| `--out` | — | также записать отчёт (leaderboard + вердикт) в этот файл |

Пример с ограниченным набором методов, тремя сидами и сохранённым отчётом:

```bash
uv run warmbench try \
  --interactions inter.csv \
  --content content.csv \
  --scores scores.csv \
  --methods linmap,stacking_plus,grouped_most_popular_pers \
  --metric auc \
  --seeds 3 \
  --out verdict.txt
```

Команда печатает тот же объект, что и `print(result)` в Python: ранжированный leaderboard методов
по выбранной метрике и однострочный вердикт, обгоняет ли трансфер baselines.

## Куда дальше

- Чем различаются методы: [Как выбрать метод](choose-method.md).
- Запустить полную воспроизводимую матрицу: [Как запустить бенчмарк](run-benchmark.md).
