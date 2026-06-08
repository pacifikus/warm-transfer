# Method families

The registry contains 17 methods. They differ mostly by which signal they transfer.

## Score-space mapping

`linmap` learns `content -> donor score vector` directly. It is model-agnostic and usually the first
strong transfer method to try. `scale_shift` starts from neighbor scores and adjusts scale/shift.

## Supervised meta-methods

`stacking`, `stacking_plus` and `logreg_calib` use a validation-cold fold. They can combine transfer
signals with popularity and affinity features, but they need `val` data.

## Content neighbors

`knn_score_avg`, `attention_knn` and `debiased_knn` use cold-to-warm similarity. They are useful
diagnostics, but naive score averaging often inherits popularity.

## Embedding methods

`linmap_emb`, `magnitude_scaling`, `embedding_avg`, `attention_emb` and `dropoutnet` need donor
embeddings. They work only when the donor exposes latent factors or item/user embeddings.

## Baselines

`random`, `most_popular`, `grouped_most_popular` and `grouped_most_popular_pers` define the floor and
the main comparison target. A transfer method that cannot beat personalized Grouped MP is probably not
useful for the studied cold-start setting.
