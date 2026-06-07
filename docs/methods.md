# Methods capability matrix

Every method is a `ColdStartMethod`: it declares `requires`, fits on `TransferInputs`, and predicts
long-format `[user_id, item_id, score]` for cold items.

## How to read the matrix

- **Requires** is the exact input kind declared by the registered class.
- **Model-agnostic** means the method needs only exported scores/interactions/content, not donor
  internals.
- **Popularity pull** marks methods that can inherit warm-neighbor popularity unless the benchmark
  proves otherwise.
- **Legend:** :white_check_mark: yes · :x: no · popularity pull :green_circle: low ·
  :yellow_circle: medium · :red_circle: high · :dart: intended (baseline by design) ·
  :heavy_minus_sign: none.

| Method | Requires | Family | Model-agnostic | Calibrates scale | Popularity pull | First try when... |
|---|---|---|---|---|---|---|
| `linmap` | `content`, `donor_scores` | score-space mapping | :white_check_mark: | :white_check_mark: | :green_circle: low | you have donor scores and item content |
| `stacking_plus` | `content`, `donor_scores`, `train_interactions`, `val` | supervised hybrid | :white_check_mark: | :white_check_mark: | :green_circle: low | a val-cold fold is available |
| `stacking` | `content`, `donor_scores`, `similarity`, `train_interactions`, `val` | supervised hybrid | :white_check_mark: | :white_check_mark: | :yellow_circle: medium | you want KNN signal plus affinity/popularity features |
| `scale_shift` | `content`, `donor_scores`, `similarity` | score calibration | :white_check_mark: | :white_check_mark: | :yellow_circle: medium | you want a scale/shift diagnostic over KNN scores |
| `logreg_calib` | `content`, `donor_scores`, `similarity`, `val` | supervised calibration | :white_check_mark: | :white_check_mark: | :yellow_circle: medium | you need a supervised diagnostic with val-cold labels |
| `knn_score_avg` | `content`, `donor_scores`, `similarity` | content neighbors | :white_check_mark: | :x: | :red_circle: high | you need the simplest neighbor-score baseline |
| `attention_knn` | `content`, `donor_scores`, `similarity` | content neighbors | :white_check_mark: | :x: | :red_circle: high | you want softmax-weighted neighbors |
| `debiased_knn` | `content`, `donor_scores`, `similarity` | content neighbors | :white_check_mark: | :x: | :yellow_circle: medium | you are testing popularity subtraction |
| `linmap_emb` | `content`, `embeddings` | embedding mapping | :x: | :white_check_mark: | :green_circle: low | the donor exposes user/item embeddings |
| `magnitude_scaling` | `content`, `embeddings` | embedding debiasing | :x: | :white_check_mark: | :green_circle: low | cold embedding norms look overconfident |
| `dropoutnet` | `content`, `embeddings` | neural embedding mapping | :x: | :white_check_mark: | :green_circle: low | you can use `torch` and donor embeddings |
| `embedding_avg` | `content`, `embeddings`, `similarity` | embedding neighbors | :x: | :x: | :red_circle: high | you need a simple embedding-neighbor baseline |
| `attention_emb` | `content`, `embeddings`, `similarity` | embedding neighbors | :x: | :x: | :yellow_circle: medium | you want attention over neighbor embeddings |
| `grouped_most_popular_pers` | `content`, `train_interactions` | baseline | :white_check_mark: | :x: | :dart: intended | you need the main strong baseline |
| `grouped_most_popular` | `content`, `train_interactions` | baseline | :white_check_mark: | :x: | :dart: intended | you need non-personalized grouped popularity |
| `most_popular` | `train_interactions` | baseline | :white_check_mark: | :x: | :dart: intended | you need global popularity |
| `random` | none | baseline | :white_check_mark: | :x: | :heavy_minus_sign: none | you need a sanity-check floor |

## Recommended path

1. Start with `grouped_most_popular_pers` and `linmap`.
2. Add `stacking_plus` when a validation-cold fold is available.
3. Use KNN and embedding-neighbor methods as diagnostics for popularity inheritance.
4. Use embedding methods only when the donor exposes embeddings.

See [Method families](explanation/methods-families.md) for the concepts and
[Main results](results/full_matrix.md) for the benchmark table.

## Important correction about calibration

Platt and isotonic calibration are monotone transformations. They can improve probability quality
(logloss/Brier), but they do not change rank-based metrics or AUC. If a method loses on AUC, the
problem is ranking, not score calibration.

## References

- SimCSR — Han & Chun, "Addressing the Item Cold-Start Using Similar Warm Items", 2021
- DropoutNet — Volkovs et al., NeurIPS 2017
- MWUF — Zhu et al., SIGIR 2021, arXiv:2105.04790
- Gantner et al., "Learning Attribute-to-Feature Mappings", ICDM 2010
- On Inherited Popularity Bias in Cold-Start, RecSys 2025, arXiv:2510.11402
