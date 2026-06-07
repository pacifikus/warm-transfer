# Зачем warm-transfer

warm-transfer нужен для item cold-start: users и donor model уже существуют, но у новых айтемов есть
контент и почти нет истории взаимодействий.

## Mental model

**warm-transfer не обучает рекомендатель.** Он берёт:

- скоры обученного донора по warm items;
- контент warm и cold items;
- optional validation-cold data, similarity или embeddings для методов, которым они нужны.

Затем он обучает `ColdStartMethod` и предсказывает скоры для cold items:

```python
method.fit(inputs, seed=42).predict(user_ids, cold_item_ids)
```

Один контракт работает для ALS, BPR, CatBoost, EASE, Two-Tower и private donor models, если вы
можете выгрузить warm-item scores.

## Когда подходит

Используйте warm-transfer, когда нужен plug&play post-processing layer, нельзя переобучать донора
после каждого нового айтема или нужен benchmark для сравнения cold-start стратегий под одним
протоколом.

Это не замена production recommender training pipeline. Это слой, который переносит сигнал уже
обученного донора на items, которых donor не видел.
