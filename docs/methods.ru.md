# Методы: матрица возможностей

Каждый метод — это `ColdStartMethod`: он объявляет `requires`, обучается на `TransferInputs` и
предсказывает long-format `[user_id, item_id, score]` для cold items.

## Как читать матрицу

- **Requires** — точные input kinds, объявленные registered class.
- **Model-agnostic** значит, что методу нужны выгруженные scores/interactions/content, а не
  внутренности донора.
- **Popularity pull** отмечает методы, которые могут наследовать popularity warm-neighbor items, если
  benchmark не докажет обратное.

| Method | Requires | Family | Model-agnostic | Calibrates scale | Popularity pull | First try when... |
|---|---|---|---|---|---|---|
| `linmap` | `content`, `donor_scores` | score-space mapping | yes | yes | low | есть donor scores и item content |
| `stacking_plus` | `content`, `donor_scores`, `train_interactions`, `val` | supervised hybrid | yes | yes | low | есть val-cold fold |
| `stacking` | `content`, `donor_scores`, `similarity`, `train_interactions`, `val` | supervised hybrid | yes | yes | medium | нужен KNN signal плюс affinity/popularity features |
| `scale_shift` | `content`, `donor_scores`, `similarity` | score calibration | yes | yes | medium | нужен scale/shift diagnostic поверх KNN scores |
| `logreg_calib` | `content`, `donor_scores`, `similarity`, `val` | supervised calibration | yes | yes | medium | нужен supervised diagnostic с val-cold labels |
| `knn_score_avg` | `content`, `donor_scores`, `similarity` | content neighbors | yes | no | high | нужен самый простой neighbor-score baseline |
| `attention_knn` | `content`, `donor_scores`, `similarity` | content neighbors | yes | no | high | нужны softmax-weighted neighbors |
| `debiased_knn` | `content`, `donor_scores`, `similarity` | content neighbors | yes | no | medium | тестируете popularity subtraction |
| `linmap_emb` | `content`, `embeddings` | embedding mapping | no | yes | low | donor отдаёт user/item embeddings |
| `magnitude_scaling` | `content`, `embeddings` | embedding debiasing | no | yes | low | cold embedding norms выглядят overconfident |
| `dropoutnet` | `content`, `embeddings` | neural embedding mapping | no | yes | low | можно использовать `torch` и donor embeddings |
| `embedding_avg` | `content`, `embeddings`, `similarity` | embedding neighbors | no | no | high | нужен простой embedding-neighbor baseline |
| `attention_emb` | `content`, `embeddings`, `similarity` | embedding neighbors | no | no | medium | нужен attention over neighbor embeddings |
| `grouped_most_popular_pers` | `content`, `train_interactions` | baseline | yes | no | intended | нужен главный сильный baseline |
| `grouped_most_popular` | `content`, `train_interactions` | baseline | yes | no | intended | нужна неперсонализированная grouped popularity |
| `most_popular` | `train_interactions` | baseline | yes | no | intended | нужна global popularity |
| `random` | none | baseline | yes | no | none | нужен sanity-check floor |

## Recommended path

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

## References

- SimCSR — Han & Chun, "Addressing the Item Cold-Start Using Similar Warm Items", 2021
- DropoutNet — Volkovs et al., NeurIPS 2017
- MWUF — Zhu et al., SIGIR 2021, arXiv:2105.04790
- Gantner et al., "Learning Attribute-to-Feature Mappings", ICDM 2010
- On Inherited Popularity Bias in Cold-Start, RecSys 2025, arXiv:2510.11402
