"""Единые имена колонок для всех DataFrame в библиотеке.

Конвенция перенята у MTS RecTools, чтобы API был идиоматичен его пользователям.
Все компоненты (методы, метрики, сплиттеры, адаптеры) обмениваются «длинными»
(long-format) DataFrame с этими именами колонок.
"""

from __future__ import annotations

from typing import Final


class Columns:
    """Namespace с фиксированными именами колонок.

    Использование::

        from warmtransfer.columns import Columns as C
        df = df.rename(columns={"uid": C.User, "iid": C.Item})
    """

    User: Final = "user_id"
    Item: Final = "item_id"
    Weight: Final = "weight"  # вес/сила взаимодействия (float; 1.0 если веса нет)
    Datetime: Final = "datetime"  # timestamp взаимодействия
    Score: Final = "score"  # скор модели (донора или cold-start метода)
    Rank: Final = "rank"  # ранг в выдаче, 1..k

    #: Минимальный набор колонок взаимодействия.
    Interactions: Final = (User, Item)
    #: Сквозной формат рекомендаций/предсказаний.
    Recommendations: Final = (User, Item, Score, Rank)
    #: Формат скоров донора (вход cold-start метода).
    Scores: Final = (User, Item, Score)
