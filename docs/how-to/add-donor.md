# Plug in a donor

A donor is any trained recommender that can score warm items. In direct library usage you can skip
adapters and pass a `[user_id, item_id, score]` table yourself. Implement `ModelAdapter` only when you
want the donor to run inside `warmbench`.

## Implement the adapter

```python
import numpy as np
import pandas as pd

from warmtransfer.bench.adapters.base import ModelAdapter, register_adapter
from warmtransfer.columns import Columns as C
from warmtransfer.types import Dataset


@register_adapter("my_donor")
class MyDonor(ModelAdapter):
    def fit(self, dataset: Dataset, seed: int = 0) -> "MyDonor":
        self._train = dataset.interactions
        return self

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                C.User: np.repeat(user_ids, len(item_ids)),
                C.Item: np.tile(item_ids, len(user_ids)),
                C.Score: 0.0,
            }
        )
```

Replace the constant score with your model inference. The returned DataFrame must contain every
requested user-item pair in long format.

## Register the module

Import the adapter from `warmtransfer.bench.adapters.__init__` so the decorator runs before
`warmbench` reads the registry.

## Use it in config

```yaml
donors:
  - name: my_donor
    params:
      some_param: 10
```

## Keep the warm-only contract

The adapter must train only on `split.train` warm interactions. It should not see test-cold items,
validation-cold labels, or benchmark metrics.
