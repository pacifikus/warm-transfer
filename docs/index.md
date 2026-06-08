# warm-transfer

Transfer and calibrate scores of a trained recommender onto cold-start items. Plug&play,
model-agnostic, no retraining.

```python
from warmtransfer.methods import LinMap

reco = LinMap(alpha=1.0).fit(inputs, seed=42).predict(user_ids, cold_item_ids)
```

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Quickstart__

    ---

    Transfer donor scores onto cold items in five minutes.

    [:octicons-arrow-right-24: Start](getting-started/quickstart.md)

-   :material-puzzle:{ .lg .middle } __Works with your scores__

    ---

    Bring any donor model's warm-item scores. warm-transfer does not retrain it.

    [:octicons-arrow-right-24: Why](explanation/why.md)

-   :material-chart-line:{ .lg .middle } __It beats strong baselines__

    ---

    Score transfer beats personalized Grouped MP in 36 of 40 dataset-donor cells.

    [:octicons-arrow-right-24: Results](results/full_matrix.md)

-   :material-book-open-variant:{ .lg .middle } __Methods & API__

    ---

    Seventeen methods, one `ColdStartMethod.fit(...).predict(...)` contract.

    [:octicons-arrow-right-24: Reference](methods.md)

</div>

## What problem it solves

Your recommender already scores warm items, but new items have no interaction history. warm-transfer
uses those warm scores plus item content to predict scores for cold-start items. The donor can be ALS,
BPR, CatBoost, EASE, a two-tower model or a private production model.

## Main result

On the current benchmark matrix of **8 dataset loaders × 5 donors** (seed=42), score-transfer methods
beat the strong personalized `grouped_most_popular_pers` baseline by **per-user AUC in 36 of 40
dataset-donor cells**.

The important contrast: naive nearest-neighbor transfer often inherits neighbor popularity and loses
to Grouped MP. Calibrated transfer, especially `linmap` and `stacking_plus`, keeps personalization from
the donor scores and is much more robust. The benchmark spans matrix-factorization, GBDT, linear
item-item and neural donors; the 4 misses are mostly on ML-1M, where the baseline AUC is already high.

These numbers are single-seed results. Targeted multi-seed runs for marginal cells are tracked in the
benchmark pages.

## Architecture

- **`warmtransfer`** is the lightweight core: transfer methods, metrics and content similarity.
- **`warmtransfer.bench`** is the optional benchmark layer: dataset loaders, splitters, donor adapters
  and the `warmbench` runner.

## Where to go next

- New to the project: start with the [Quickstart](getting-started/quickstart.md).
- Bringing a trained model: read [Plug in a donor](how-to/add-donor.md).
- Choosing between methods: use the [capability matrix](methods.md).
- Checking scientific validity: read the [evaluation protocol](eval-protocol.md).
