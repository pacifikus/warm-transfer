# Choose a method

Start from the inputs you can provide. The method registry validates `requires`, so choosing a method
is mostly a data-availability question.

## Decision table

| You have... | Try first | Then compare with |
|---|---|---|
| Warm donor scores + item content | `linmap` | `grouped_most_popular_pers`, `knn_score_avg` |
| Warm donor scores + content + val-cold fold | `stacking_plus` | `linmap`, `stacking` |
| Cold-to-warm similarity only | `knn_score_avg` | `attention_knn`, `debiased_knn` |
| Donor embeddings | `linmap_emb` | `attention_emb`, `magnitude_scaling`, `dropoutnet` |
| Only interactions/content, no donor scores | `grouped_most_popular_pers` | `most_popular` |

## Recommended benchmark set

```yaml
methods:
  - name: grouped_most_popular_pers
  - name: linmap
  - name: stacking_plus
  - name: knn_score_avg
```

Add embedding methods only for donors that expose embeddings. CatBoost and EASE do not provide the
same latent-factor interface as ALS/BPR/Two-Tower, so `[EMB]` methods can be skipped for them.

## Interpret the result

- If `linmap` beats Grouped MP, the donor score structure transfers through content.
- If `stacking_plus` wins, validation-cold data adds useful popularity/affinity features.
- If KNN methods win, inspect whether the gain is real personalization or inherited popularity.
- If no transfer method beats Grouped MP, the cold content may not explain the donor signal.
