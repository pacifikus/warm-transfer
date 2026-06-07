# Датасеты

Требование к датасету для cold-start item бенчмарка: нужны **новые айтемы** (можно эмулировать)
и **контентные признаки айтема** (жанры/категории/текст), чтобы оценивать рекомендации по
контенту без истории взаимодействий.

| Приоритет | Датасет | Домен | Размер (users / items / interactions) | Контент-фичи айтема | Feedback | Загрузка |
|---|---|---|---|---|---|---|
| 1 | **MTS KION** | фильмы/сериалы (RU) | 962K / 15.7K / 5.48M | жанры, страны, студии, описания (RU+EN) | implicit (watch %) | `github.com/irsafilo/KION_DATASET` (.zip CSV) |
| 2 | **MovieLens-1M** | кино | 6K / 3.9K / 1M | жанры, год | explicit (ratings) | grouplens (.zip) — внешний бейзлайн GroupedMP AUC≈0.709 |
| 3 | **Goodbooks-10k** | книги | 53K / 10K / 6M | авторы, жанры (tags), год | explicit | zygmuntz/goodbooks-10k |
| 4 | **Amazon subset** (Toys/Electronics) | e-com | варьируется (урезаем) | категории, бренд, TF-IDF(title/desc) | explicit (ratings) | Amazon Reviews (McAuley) |
| опц. | **Yambda-50M** | музыка (RU) | ~1M / млн / 50M | audio-эмбеддинги, artist/album | implicit + explicit | HuggingFace `yandex/yambda` |

Покрываемые оси обобщаемости: домены (кино/книги/e-com/музыка), explicit/implicit feedback,
тип контента (категориальный / текст / эмбеддинги), масштаб (1M…50M), язык (RU/EN).

## Статус реализации

- ✅ **MovieLens-1M** (`ml-1m`) — реализован, основной для сверки и анализа.
- ✅ **MovieLens-20M** (`ml-20m`) — реализован, большой кейс кино (20M рейтингов) для масштаба.
- ✅ **Goodbooks-10k** (`goodbooks`) — реализован, второй домен (книги).
- ✅ **MTS KION** (`kion`) — реализован, третий домен (RU, фильмы/сериалы, implicit).
- ✅ **KION-text** (`kion-text`) — тот же KION, контент = TF-IDF текста (для абляции контента).
- ⬜ Amazon, Yambda — в планах (загрузчики не написаны).

Доноры: ✅ ALS (`als`), ✅ CatBoost (`catboost`), ✅ BPR (`bpr`). Бенчмарк прогнан на
3 датасетах × 3 донора (см. `results/full_matrix.md`).

## Память (M1 16GB)

KION, ML-1M, Goodbooks грузятся целиком. Amazon и Yambda — субсэмплируем (по пользователям/
времени) до влезающего размера. Скоры доноров кэшируются в parquet, плотные матрицы
users×items не материализуем.

## Контракт загрузчика

Каждый датасет реализует `DatasetLoader.load() -> Dataset` (interactions long-format +
`ItemFeatures`). Добавление нового датасета = один класс + декоратор `@register_dataset(...)`.
