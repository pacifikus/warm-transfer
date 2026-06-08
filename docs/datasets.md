# Datasets

Requirement for a dataset used in the cold-start item benchmark: it needs **new items** (can be emulated)
and **item content features** (genres/categories/text) in order to evaluate recommendations by
content without interaction history.

| Priority | Dataset | Domain | Size (users / items / interactions) | Item content features | Feedback | Loading |
|---|---|---|---|---|---|---|
| 1 | **MTS KION** | movies/series (RU) | 962K / 15.7K / 5.48M | genres, countries, studios, descriptions (RU+EN) | implicit (watch %) | `github.com/irsafilo/KION_DATASET` (.zip CSV) |
| 2 | **MovieLens-1M** | cinema | 6K / 3.9K / 1M | genres, year | explicit (ratings) | grouplens (.zip) — external baseline GroupedMP AUC≈0.709 (multi-seed mean 0.7082±0.007 over 5 seeds, see `results/seeds.md`; cf. single-seed 0.7202 in `results/full_matrix.md`) |
| 3 | **Goodbooks-10k** | books | 53K / 10K / 6M | authors, genres (tags), year | explicit | zygmuntz/goodbooks-10k |
| 4 | **Amazon Toys & Games** (5-core) | e-com | 19.4K / 11.9K / 168K | categories, brand, TF-IDF(title/desc) | explicit (ratings) | SNAP 5-core (McAuley) |
| 5 | **MIND** (MINDlarge) | news (EN) | 750K / 19.2K / 3.93M | category, subcategory, TF-IDF(title/abstract) | implicit (clicks) | HuggingFace `yjw1029/MIND` |
| opt. | **Yambda-50M** | music (RU) | ~1M / millions / 50M | audio embeddings, artist/album | implicit + explicit | HuggingFace `yandex/yambda` |

Generalizability axes covered: domains (cinema/books/e-com/music), explicit/implicit feedback,
content type (categorical / text / embeddings), scale (1M…50M), language (RU/EN).

## Implementation status

- ✅ **MovieLens-1M** (`ml-1m`) — implemented, the primary one for verification and analysis.
- ✅ **MovieLens-20M** (`ml-20m`) — implemented, a large cinema case (20M ratings) for scale.
- ✅ **Goodbooks-10k** (`goodbooks`) — implemented, the second domain (books).
- ✅ **MTS KION** (`kion`) — implemented, the third domain (RU, movies/series, implicit).
- ✅ **KION-text** (`kion-text`) — the same KION, content = TF-IDF of text (for content ablation).
- ✅ **Amazon Toys** (`amazon-toys`) — implemented, e-com (5-core), content = top categories + brand.
- ✅ **Amazon Toys-text** (`amazon-toys-text`) — the same Amazon, content = TF-IDF of title + desc.
- ✅ **MIND** (`mind`) — implemented, news (EN, implicit clicks), content = category + subcategory.
- ✅ **MIND-text** (`mind-text`) — the same MIND, content = TF-IDF of title + abstract.
- ⬜ Yambda — planned (loader not written yet).

Donors: ✅ ALS (`als`), ✅ BPR (`bpr`), ✅ CatBoost (`catboost`), ✅ EASE (`ease`),
✅ Two-Tower (`two_tower`). The benchmark was run on 8 dataset loaders × 5 donors
(see `results/full_matrix.md`).

## Memory (M1 16GB)

KION, ML-1M, Goodbooks are loaded in full. Amazon and Yambda are subsampled (by users/
time) down to a size that fits. Donor scores are cached in parquet; dense
users×items matrices are not materialized.

## Loader contract

Each dataset implements `DatasetLoader.load() -> Dataset` (interactions long-format +
`ItemFeatures`). Adding a new dataset = one class + the `@register_dataset(...)` decorator.
