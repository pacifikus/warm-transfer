# warm-transfer

**Model-agnostic plug&play библиотека** для трансфера и калибровки скоров уже обученной
рекомендательной модели на **новые (cold-start) айтемы** в условиях экстремальной
разреженности, плюс воспроизводимый **бенчмарк**.

Идея: у вас есть обученная модель произвольной архитектуры — библиотека «накладывается»
поверх её скоров и оценивает новые товары/контент, для которых ещё нет (или почти нет)
взаимодействий. Модель переобучать не нужно, доступа к её внутренностям тоже.

## Структура

- **`warmtransfer`** — лёгкое ядро (plug&play). Работает со скорами донора + контентом.
  Ставится без тяжёлых recsys-зависимостей.
  - `methods/` — cold-start методы (бейзлайны, KNN, LinMap, Stacking, scale&shift, attention-KNN…)
  - `metrics/` — собственные корректные метрики (Recall/Precision/MAP/NDCG@k, MRR, AUC, RelaImpr)
  - `similarity/` — контентное сходство cold→warm (опционально)
- **`warmtransfer.bench`** — бенчмарк (тяжёлые зависимости, extra `bench`).
  - `datasets/` — загрузчики (ML-1M/20M, Goodbooks, KION, KION-text)
  - `splitters/` — честный pseudo-cold сплит (анти-утечка)
  - `adapters/` — модели-доноры (ALS, BPR, CatBoost)
  - `runner.py` — прогон матрицы датасеты × доноры × методы × бейзлайны

## Ключевой результат

Model-agnostic методы **LinMap** (Ridge: контент → вектор скоров донора) и **stacking_plus**
(гибрид: linmap-сигнал + персонализированная популярность) **обходят сильный персонализированный
Grouped MP** на полной матрице **3 домена × 3 донора**: по AUC в 7 из 9 ячеек, по ранжированию
на ML-1M и KION со всеми донорами. Разрывы превышают разброс по 5 сидам. Наивные методы
(knn/attention/debiased/embedding_avg) проигрывают бейзлайну — тянут популярность соседей.
DropoutNet (deep [EMB]) даёт лучший ranking на плотном ML-1M. Подробности и таблицы —
`docs/results/full_matrix.md`.

## Plug&play использование (ядро, без бенчмарка)

```python
import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.methods import LinMap
from warmtransfer.types import ItemFeatures, TransferInputs

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

# приносите свои warm-скоры донора + контент warm/cold айтемов
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
uv sync --extra bench   # + движки доноров и инфраструктура бенчмарка
uv sync --extra all     # + deep (torch)
```

## Запуск

```bash
uv run pytest -q                          # тесты ядра
uv run python examples/quickstart.py      # минимальный plug&play пример
uv run warmbench --list-components        # доступные datasets/donors/methods
uv run warmbench --config configs/example.yaml --dry-run
uv run warmbench --config configs/example.yaml  # пример прогона бенчмарка
```

## Документация

Сайт документации собирается через MkDocs Material (как у RecTools) и публикуется на
GitHub Pages автоматически (`.github/workflows/docs.yml`) после пуша в GitHub-репозиторий.
Локально:

```bash
uv sync --group docs
uv run mkdocs serve     # локальный предпросмотр на http://127.0.0.1:8000
uv run mkdocs build     # статический сайт в site/
```

Исходники страниц:
- `docs/methods.md` — описание методов
- `docs/datasets.md` — описание датасетов
- `docs/eval-protocol.md` — протокол сплита и метрик (анти-утечка)
- `docs/results/` — таблицы результатов (полная матрица, разброс по сидам, абляции)
- `docs/api.md` — авто-справочник API
