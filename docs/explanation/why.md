# Why warm-transfer

warm-transfer is for the item cold-start case: users and the donor model already exist, but new
items have content and little or no interaction history.

## Mental model

**warm-transfer does not train a recommender.** It takes:

- scores from a trained donor over warm items;
- content for warm and cold items;
- optional validation-cold data, similarity or embeddings for methods that require them.

Then it fits a `ColdStartMethod` and predicts scores for cold items:

```python
method.fit(inputs, seed=42).predict(user_ids, cold_item_ids)
```

The same contract works for ALS, BPR, CatBoost, EASE, Two-Tower and private donor models, as long as
you can export warm-item scores.

## When it fits

Use warm-transfer when you need a plug&play post-processing layer, cannot retrain the donor on every
new item, or want a benchmark that compares cold-start strategies under one protocol.

It is not a replacement for a production recommender training pipeline. It is the layer that transfers
the trained donor's signal onto items that the donor never saw.
