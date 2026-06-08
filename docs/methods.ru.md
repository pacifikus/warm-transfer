# Методы: матрица возможностей

Каждый метод — это `ColdStartMethod`: он объявляет `requires`, обучается на `TransferInputs` и
предсказывает long-format `[user_id, item_id, score]` для cold items.

## Как читать матрицу

- **Requires** — точные input kinds, объявленные registered class.
- **Model-agnostic** значит, что методу нужны выгруженные scores/interactions/content, а не
  внутренности донора.
- **Popularity pull** отмечает методы, которые могут наследовать popularity warm-neighbor items, если
  benchmark не докажет обратное.
- **Обозначения:** :white_check_mark: да · :x: нет · popularity pull :green_circle: low ·
  :yellow_circle: medium · :red_circle: high · :dart: intended (у baseline by design) ·
  :heavy_minus_sign: none.

| Method | Requires | Family | Model-agnostic | Calibrates scale | Popularity pull | First try when... |
|---|---|---|---|---|---|---|
| `linmap` | `content`, `donor_scores` | score-space mapping | :white_check_mark: | :white_check_mark: | :green_circle: low | есть donor scores и item content |
| `stacking_plus` | `content`, `donor_scores`, `train_interactions`, `val` | supervised hybrid | :white_check_mark: | :white_check_mark: | :green_circle: low | есть val-cold fold |
| `stacking` | `content`, `donor_scores`, `similarity`, `train_interactions`, `val` | supervised hybrid | :white_check_mark: | :white_check_mark: | :yellow_circle: medium | нужен KNN signal плюс affinity/popularity features |
| `scale_shift` | `content`, `donor_scores`, `similarity` | score-space mapping | :white_check_mark: | :white_check_mark: | :yellow_circle: medium | нужен scale/shift diagnostic поверх KNN scores |
| `logreg_calib` | `content`, `donor_scores`, `similarity`, `val` | supervised calibration | :white_check_mark: | :white_check_mark: | :yellow_circle: medium | нужен supervised diagnostic с val-cold labels |
| `knn_score_avg` | `content`, `donor_scores`, `similarity` | content neighbors | :white_check_mark: | :x: | :red_circle: high | нужен самый простой neighbor-score baseline |
| `attention_knn` | `content`, `donor_scores`, `similarity` | content neighbors | :white_check_mark: | :x: | :red_circle: high | нужны softmax-weighted neighbors |
| `debiased_knn` | `content`, `donor_scores`, `similarity` | content neighbors | :white_check_mark: | :x: | :yellow_circle: medium | тестируете popularity subtraction |
| `linmap_emb` | `content`, `embeddings` | embedding mapping | :x: | :white_check_mark: | :green_circle: low | donor отдаёт user/item embeddings |
| `magnitude_scaling` | `content`, `embeddings` | embedding debiasing | :x: | :white_check_mark: | :green_circle: low | cold embedding norms выглядят overconfident |
| `dropoutnet` | `content`, `embeddings` | neural embedding mapping | :x: | :white_check_mark: | :green_circle: low | можно использовать `torch` и donor embeddings |
| `embedding_avg` | `content`, `embeddings`, `similarity` | embedding neighbors | :x: | :x: | :red_circle: high | нужен простой embedding-neighbor baseline |
| `attention_emb` | `content`, `embeddings`, `similarity` | embedding neighbors | :x: | :x: | :yellow_circle: medium | нужен attention over neighbor embeddings |
| `grouped_most_popular_pers` | `content`, `train_interactions` | baseline | :white_check_mark: | :x: | :dart: intended | нужен главный сильный baseline |
| `grouped_most_popular` | `content`, `train_interactions` | baseline | :white_check_mark: | :x: | :dart: intended | нужна неперсонализированная grouped popularity |
| `most_popular` | `train_interactions` | baseline | :white_check_mark: | :x: | :dart: intended | нужна global popularity |
| `random` | none | baseline | :white_check_mark: | :x: | :heavy_minus_sign: none | нужен sanity-check floor |

## Рекомендуемый путь

1. Начните с `grouped_most_popular_pers` и `linmap`.
2. Добавьте `stacking_plus`, если есть validation-cold fold.
3. Используйте KNN и embedding-neighbor methods как diagnostics для popularity inheritance.
4. Используйте embedding methods только если donor отдаёт embeddings.

См. [Семейства методов](explanation/methods-families.md) для концепций и
[главные результаты](results/full_matrix.md) для benchmark table.

## Важная поправка про calibration

Platt и isotonic calibration — монотонные преобразования. Они могут улучшить probability quality
(logloss/Brier), но не меняют rank-based metrics и AUC. Если метод проигрывает по AUC, проблема в
ranking, а не в score calibration.

## Ссылки

- SimCSR — Han & Chun, "Addressing the Item Cold-Start Using Similar Warm Items", 2021
- DropoutNet — Volkovs et al., NeurIPS 2017
- MWUF — Zhu et al., SIGIR 2021, arXiv:2105.04790
- Gantner et al., "Learning Attribute-to-Feature Mappings", ICDM 2010
- On Inherited Popularity Bias in Cold-Start, RecSys 2025, arXiv:2510.11402
