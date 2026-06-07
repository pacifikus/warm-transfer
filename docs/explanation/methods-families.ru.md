# Семейства методов

В registry 17 методов. Они отличаются прежде всего тем, какой сигнал переносят.

## Score-space mapping

`linmap` учит `content -> donor score vector` напрямую. Это model-agnostic метод и обычно первый
сильный transfer candidate. `scale_shift` начинает с neighbor scores и корректирует scale/shift.

## Supervised meta-methods

`stacking`, `stacking_plus` и `logreg_calib` используют validation-cold fold. Они могут сочетать
transfer signals с popularity и affinity features, но требуют `val` data.

## Content neighbors

`knn_score_avg`, `attention_knn` и `debiased_knn` используют cold-to-warm similarity. Они полезны как
diagnostics, но наивное score averaging часто наследует popularity.

## Embedding methods

`linmap_emb`, `magnitude_scaling`, `embedding_avg`, `attention_emb` и `dropoutnet` требуют donor
embeddings. Они работают только когда donor отдаёт latent factors или user/item embeddings.

## Baselines

`random`, `most_popular`, `grouped_most_popular` и `grouped_most_popular_pers` задают floor и главный
target для сравнения. Transfer method, который не обходит personalized Grouped MP, скорее всего, не
полезен для изучаемого cold-start setting.
