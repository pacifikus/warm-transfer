# Как добавить свой метод

Transfer method получает `TransferInputs`, объявляет нужные поля и предсказывает scores для cold
items.

## Минимальный метод

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

## Объявите inputs

Поставьте в `requires` поля, которые реально нужны методу:

- `donor_scores`
- `content`
- `similarity`
- `embeddings`
- `train_interactions`
- `item_meta`
- `val`

Base class проверит их до `_fit`.

## Сделайте метод discoverable

Импортируйте module из `warmtransfer.methods.__init__`, чтобы `@register_method(...)` выполнился.
Затем добавьте unit test, который fit-ит метод на маленьком `TransferInputs` fixture и проверяет
output columns и shape.

## Используйте в benchmark config

```yaml
methods:
  - name: constant_zero
```
