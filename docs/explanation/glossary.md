# Glossary

| Term | Meaning |
|---|---|
| Donor | A trained recommender whose scores over warm items are transferred to cold items. |
| Warm item | An item available during donor training. |
| Cold item | An item hidden from donor training and evaluated as a new item. |
| Pseudo-cold | A warm historical item deliberately held out to emulate cold-start evaluation. |
| `TransferInputs` | The bundle passed to `ColdStartMethod.fit`: scores, content, similarity, embeddings and optional val data. |
| `requires` | Method-declared input kinds that are validated before fitting. |
| `ModelAdapter` | Benchmark wrapper around a donor engine. It trains on warm interactions and emits scores. |
| `DatasetLoader` | Benchmark component that loads interactions and item content into a `Dataset`. |
| `Splitter` | Benchmark component that creates warm, val-cold and test-cold folds. |
| Grouped MP | Grouped Most Popular baseline; popularity is computed within the cold item's content group. |
| Personalized Grouped MP | Strong baseline using the user's affinity to the cold item's groups. |
| Score transfer | Learning how content maps to donor scores, then applying that mapping to cold items. |
