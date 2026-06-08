# Как подключить донора

Donor — это любой обученный рекомендатель, который может скорить warm items. При прямом использовании
библиотеки adapter не нужен: можно передать таблицу `[user_id, item_id, score]` самостоятельно.
`ModelAdapter` нужен, когда donor должен запускаться внутри `warmbench`.

## Реализуйте adapter

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

Замените constant score на inference вашей модели. Возвращаемый DataFrame должен содержать все
запрошенные пары user-item в long format.

## Зарегистрируйте module

Импортируйте adapter из `warmtransfer.bench.adapters.__init__`, чтобы decorator выполнился до чтения
registry в `warmbench`.

## Используйте в config

```yaml
donors:
  - name: my_donor
    params:
      some_param: 10
```

## Сохраните warm-only contract

Adapter должен обучаться только на `split.train` warm interactions. Он не должен видеть test-cold
items, validation-cold labels или benchmark metrics.
