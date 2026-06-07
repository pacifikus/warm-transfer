# Datasets

Requirement for a dataset used in the cold-start item benchmark: it needs **new items** (can be emulated)
and **item content features** (genres/categories/text) in order to evaluate recommendations by
content without interaction history.

| Priority | Dataset | Domain | Size (users / items / interactions) | Item content features | Feedback | Loading |
|---|---|---|---|---|---|---|
| 1 | **MTS KION** | movies/series (RU) | 962K / 15.7K / 5.48M | genres, countries, studios, descriptions (RU+EN) | implicit (watch %) | `github.com/irsafilo/KION_DATASET` (.zip CSV) |
| 2 | **MovieLens-1M** | cinema | 6K / 3.9K / 1M | genres, year | explicit (ratings) | grouplens (.zip) — external baseline GroupedMP AUC≈0.709 |
| 3 | **Goodbooks-10k** | books | 53K / 10K / 6M | authors, genres (tags), year | explicit | zygmuntz/goodbooks-10k |
| 4 | **Amazon subset** (Toys/Electronics) | e-com | varies (subsampled) | categories, brand, TF-IDF(title/desc) | explicit (ratings) | Amazon Reviews (McAuley) |
| opt. | **Yambda-50M** | music (RU) | ~1M / millions / 50M | audio embeddings, artist/album | implicit + explicit | HuggingFace `yandex/yambda` |

Generalizability axes covered: domains (cinema/books/e-com/music), explicit/implicit feedback,
content type (categorical / text / embeddings), scale (1M…50M), language (RU/EN).

## Implementation status

- ✅ **MovieLens-1M** (`ml-1m`) — implemented, the primary one for verification and analysis.
- ✅ **MovieLens-20M** (`ml-20m`) — implemented, a large cinema case (20M ratings) for scale.
- ✅ **Goodbooks-10k** (`goodbooks`) — implemented, the second domain (books).
- ✅ **MTS KION** (`kion`) — implemented, the third domain (RU, movies/series, implicit).
- ✅ **KION-text** (`kion-text`) — the same KION, content = TF-IDF of text (for content ablation).
- ⬜ Amazon, Yambda — planned (loaders not written yet).

Donors: ✅ ALS (`als`), ✅ CatBoost (`catboost`), ✅ BPR (`bpr`). The benchmark was run on
3 datasets × 3 donors (see `results/full_matrix.md`).

## Memory (M1 16GB)

KION, ML-1M, Goodbooks are loaded in full. Amazon and Yambda are subsampled (by users/
time) down to a size that fits. Donor scores are cached in parquet; dense
users×items matrices are not materialized.

## Loader contract

Each dataset implements `DatasetLoader.load() -> Dataset` (interactions long-format +
`ItemFeatures`). Adding a new dataset = one class + the `@register_dataset(...)` decorator.
