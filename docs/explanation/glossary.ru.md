# Глоссарий

| Термин | Значение |
|---|---|
| Donor | Обученный рекомендатель, чьи scores по warm items переносятся на cold items. |
| Warm item | Item, доступный во время donor training. |
| Cold item | Item, скрытый от donor training и оцениваемый как новый. |
| Pseudo-cold | Исторический warm item, специально отложенный для эмуляции cold-start evaluation. |
| `TransferInputs` | Bundle для `ColdStartMethod.fit`: scores, content, similarity, embeddings и optional val data. |
| `requires` | Method-declared input kinds, которые проверяются перед fitting. |
| `ModelAdapter` | Benchmark wrapper вокруг donor engine. Обучается на warm interactions и отдаёт scores. |
| `DatasetLoader` | Benchmark component, который грузит interactions и item content в `Dataset`. |
| `Splitter` | Benchmark component, который создаёт warm, val-cold и test-cold folds. |
| Grouped MP | Grouped Most Popular baseline; popularity считается внутри content group cold item. |
| Personalized Grouped MP | Сильный baseline, который использует affinity пользователя к группам cold item. |
| Score transfer | Обучение отображения из content в donor scores и применение этого отображения к cold items. |
