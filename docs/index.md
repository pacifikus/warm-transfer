# coldscore + coldbench

**Model-agnostic plug&play библиотека** для переноса и калибровки скоров уже обученной
рекомендательной модели на **новые (cold-start) айтемы** при экстремальной разреженности,
плюс воспроизводимый **бенчмарк**.

Идея: у вас есть обученная модель произвольной архитектуры — библиотека «накладывается»
поверх её скоров и оценивает новые товары/контент, для которых ещё нет (или почти нет)
взаимодействий. Модель переобучать не нужно, доступа к её внутренностям тоже.

## Главный результат

Model-agnostic методы **LinMap** (Ridge: контент → вектор скоров донора) и **stacking_plus**
(гибрид: linmap-сигнал + персонализированная популярность) **обходят сильный персонализированный
Grouped MP** на полной матрице **3 домена × 3 донора**:

- по **AUC** — в 7 из 9 ячеек (с донором ALS — во всех трёх доменах);
- по **ранжированию** (NDCG@10, Recall@10) — на ML-1M и KION со всеми донорами;
- разрывы превышают разброс по 5 сидам.

Наивные методы (KNN-усреднение, attention, embedding-avg) проигрывают бейзлайну — тянут
популярность соседей. Подробности и таблицы — в разделе [Результаты](results/full_matrix.md).

## Архитектура

- **`coldscore`** — лёгкое ядро (plug&play): методы переноса, метрики, similarity.
  Работает со скорами донора + контентом, ставится без тяжёлых recsys-зависимостей.
- **`coldbench`** — бенчмарк (extra `bench`): датасеты, честный сплиттер, доноры
  (ALS/BPR/CatBoost), runner.

## Быстрый старт (ядро)

```python
import numpy as np
import pandas as pd

from coldscore.columns import Columns as C
from coldscore.methods import LinMap
from coldscore.types import ItemFeatures, TransferInputs

warm_features = ItemFeatures(
    item_ids=np.array([10, 11]),
    matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
    feature_names=["genre_action", "genre_drama"],
)
cold_features = ItemFeatures(
    item_ids=np.array([20]),
    matrix=np.array([[1.0, 0.0]]),
    feature_names=["genre_action", "genre_drama"],
)
donor_scores = pd.DataFrame(
    {
        C.User: [1, 1, 2, 2],
        C.Item: [10, 11, 10, 11],
        C.Score: [5.0, 1.0, 1.0, 5.0],
    }
)

inputs = TransferInputs(
    donor_scores=donor_scores,
    warm_features=warm_features,
    cold_features=cold_features,
)
reco = LinMap(alpha=1.0).fit(inputs, seed=42).predict(
    user_ids=np.array([1, 2]),
    cold_item_ids=np.array([20]),
)
```

Полный исполняемый пример: `examples/quickstart.py`.

## Установка

```bash
uv sync                 # только ядро + dev
uv sync --extra bench   # + движки доноров и бенчмарк
uv sync --extra all     # + deep (torch)
```

## Проверка бенчмарка

```bash
uv run python examples/quickstart.py
uv run coldbench --list-components
uv run coldbench --config configs/example.yaml --dry-run
uv run coldbench --config configs/example.yaml
```

См. [Методы](methods.md), [Датасеты](datasets.md), [Протокол оценки](eval-protocol.md),
[API](api.md).
