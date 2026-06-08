# Как запустить бенчмарк

`warmbench` запускает datasets × donors × methods под единым pseudo-cold protocol.

## Посмотреть доступные компоненты

```bash
uv run warmbench --list-components
```

## Начать с example config

```bash
uv run warmbench --config configs/example.yaml --dry-run
uv run warmbench --config configs/example.yaml
```

Example запускает ML-1M × ALS на baselines и `knn_score_avg`.

## Флаги основной команды

Плоская команда `warmbench` (без подкоманды) принимает следующие флаги:

| Флаг | Дефолт | Значение |
|---|---|---|
| `--config` | — | путь к YAML config прогона (обязателен, кроме режима `--list-components`) |
| `--name` | `results` | имя выходной таблицы |
| `--std` | off | добавить в таблицу результатов колонки std по сидам |
| `--rela-base` | — | baseline-метод для колонки `RelaImpr` (относительный прирост AUC) |
| `--dry-run` | off | проверить config и зарегистрированные компоненты без запуска |
| `--list-components` | off | напечатать зарегистрированные datasets/donors/methods/splitters |

Примеры:

```bash
# добавить колонки std по сидам и своё имя таблицы
uv run warmbench --config configs/example.yaml --std --name als_run

# добавить колонку RelaImpr относительно baseline Grouped MP
uv run warmbench --config configs/example.yaml --rela-base grouped_most_popular_pers
```

## `warmbench try`: быстрый вердикт по своим данным

`warmbench try` оценивает методы на ваших данных и печатает leaderboard + вердикт, без полной
матрицы dataset/donor. Это зеркало `warmtransfer.recommend()`. Полный разбор — в отдельном рецепте
[Быстрый вердикт](quick-verdict.md).

```bash
uv run warmbench try \
  --interactions inter.parquet \
  --content content.parquet \
  --scores scores.parquet
```

Обязательные входы (Parquet, если путь оканчивается на `.parquet`, иначе CSV):

| Флаг | Колонки |
|---|---|
| `--interactions` | `user_id, item_id` (тёплая история) |
| `--content` | `item_id` + числовые колонки признаков |
| `--scores` | `user_id, item_id, score` (скоры донора по тёплым айтемам) |

Опции:

| Флаг | Дефолт | Значение |
|---|---|---|
| `--metric` | `auc` | основная метрика для ранжирования и вердикта |
| `--methods` | `all` | имена методов через запятую или `all` |
| `--seeds` | `1` | сколько сидов усреднять |
| `--seed` | `42` | базовый random seed |
| `--out` | — | записать отчёт (leaderboard + вердикт) в этот файл |

Команда печатает ранжированный по выбранной метрике leaderboard и однострочный вердикт о том,
обгоняет ли трансфер baselines.

## Форма config

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
    label: linmap_a2          # тот же метод, другие гиперпараметры, отдельная строка результатов
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

Два дополнительных ключа стоит знать:

- **`label`** (на компонент, опционально) — имя, показываемое в таблице результатов; по умолчанию
  равно `name`. Используйте для ablation: перечислите один метод несколько раз с разными `params`
  под разными label, чтобы каждый стал отдельной строкой в одном прогоне.
- **`max_eval_users`** (верхний уровень, дефолт `2000`) — ограничивает, сколько пользователей
  скорится при оценке. Уменьшайте для быстрых smoke-прогонов; задайте `null`, чтобы оценивать
  всех пользователей.

## Читать outputs

Runner пишет Markdown/CSV/Parquet artifacts в `out_dir`. Основные таблицы в `docs/results/`
собираются из этих benchmark artifacts.

## Guardrails

- Запускайте `--dry-run` после изменения YAML.
- Держите `seeds` явными.
- Используйте одинаковые splitter и metric settings при сравнении методов.
- Считайте single-seed results directional, пока не проверен multi-seed variance.
