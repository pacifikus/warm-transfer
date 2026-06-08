# Add your own method

A transfer method receives `TransferInputs`, declares what fields it needs, and predicts scores for
cold items.

## Minimal method

```python
import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.methods.base import ColdStartMethod, register_method
from warmtransfer.types import TransferInputs


@register_method("constant_zero")
class ConstantZero(ColdStartMethod):
    requires = frozenset()

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        return None

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        return pd.DataFrame(
            {
                C.User: np.repeat(user_ids, len(cold_item_ids)),
                C.Item: np.tile(cold_item_ids, len(user_ids)),
                C.Score: 0.0,
            }
        )
```

## Declare inputs

Set `requires` to the fields your method actually needs:

- `donor_scores`
- `content`
- `similarity`
- `embeddings`
- `train_interactions`
- `item_meta`
- `val`

The base class validates them before `_fit` runs.

## Make it discoverable

Import the module from `warmtransfer.methods.__init__` so `@register_method(...)` runs. Then add a
unit test that fits the method on a tiny `TransferInputs` fixture and verifies the output columns and
shape.

## Use it in benchmark config

```yaml
methods:
  - name: constant_zero
```
