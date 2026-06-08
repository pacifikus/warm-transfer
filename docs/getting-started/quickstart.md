# Quickstart

This page shows the core plug&play path: bring donor scores over warm items, fit one transfer
method, and predict scores for cold-start items. It does not use `warmtransfer.bench`.

## Install

=== "uv"

    ```bash
    uv sync
    ```

=== "pip"

    ```bash
    python -m pip install warm-transfer
    ```

## Run the example

The full runnable script lives in the repository and is included here directly, so the docs cannot
drift away from the checked example.

```python
--8<-- "examples/quickstart.py"
```

Expected output:

```text
 user_id  item_id  score
       1       20    4.0
       2       20    2.0
```

## What happened

1. `donor_scores` is a long-format table `[user_id, item_id, score]` over warm items only.
2. `warm_features` and `cold_features` align item ids with content vectors.
3. `LinMap.fit(inputs, seed=42)` learns a linear map from item content to a vector of donor scores.
4. `predict(user_ids, cold_item_ids)` returns long-format scores for all requested user-item pairs.

## Next steps

- Need package variants? Read [Installation](installation.md).
- Choosing another transfer method? Use the [capability matrix](../methods.md).
- Adding your own method? Follow [Add your own method](../how-to/add-method.md).
- Running the benchmark? Follow [Run the benchmark](../how-to/run-benchmark.md).
