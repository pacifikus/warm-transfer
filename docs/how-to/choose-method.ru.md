# Как выбрать метод

Начните с inputs, которые можете предоставить. Method registry проверяет `requires`, поэтому выбор
метода в основном зависит от доступности данных.

## Decision table

| У вас есть... | Try first | Then compare with |
|---|---|---|
| Warm donor scores + item content | `linmap` | `grouped_most_popular_pers`, `knn_score_avg` |
| Warm donor scores + content + val-cold fold | `stacking_plus` | `linmap`, `stacking` |
| Cold-to-warm similarity only | `knn_score_avg` | `attention_knn`, `debiased_knn` |
| Donor embeddings | `linmap_emb` | `attention_emb`, `magnitude_scaling`, `dropoutnet` |
| Только interactions/content, без donor scores | `grouped_most_popular_pers` | `most_popular` |

## Recommended benchmark set

```yaml
methods:
  - name: grouped_most_popular_pers
  - name: linmap
  - name: stacking_plus
  - name: knn_score_avg
```

Добавляйте embedding methods только для donors, которые отдают embeddings. CatBoost и EASE не дают
такой же latent-factor interface, как ALS/BPR/Two-Tower, поэтому `[EMB]` methods для них можно
пропускать.

## Интерпретировать результат

- Если `linmap` бьёт Grouped MP, donor score structure переносится через content.
- Если выигрывает `stacking_plus`, validation-cold data добавляет полезные popularity/affinity features.
- Если выигрывают KNN methods, проверьте, это real personalization или inherited popularity.
- Если ни один transfer method не бьёт Grouped MP, cold content может не объяснять donor signal.
