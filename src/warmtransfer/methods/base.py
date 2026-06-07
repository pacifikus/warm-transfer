"""Базовый интерфейс cold-start метода (трансфер/калибровка скоров) + реестр.

Это ядро plug&play: метод принимает скоры донора по warm-айтемам (+опц. контент/
сходство/эмбеддинги) и предсказывает скоры для пар (user, cold_item).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.exceptions import MissingInputError, NotFittedError
from warmtransfer.registry import Registry
from warmtransfer.types import TransferInputs

#: Допустимые источники входа, которые метод может объявить в ``requires``.
INPUT_KINDS = frozenset(
    {
        "donor_scores",
        "train_interactions",
        "content",
        "similarity",
        "embeddings",
        "item_meta",
        "val",  # val-cold фолд (для супервизорных методов)
    }
)


class ColdStartMethod(ABC):
    """Абстрактный cold-start метод.

    Подклассы объявляют:
      * ``name`` — имя для реестра/конфига;
      * ``requires`` — какие входы обязательны (валидируется в :meth:`fit`).

    Контракт:
      * :meth:`fit` возвращает ``self``;
      * :meth:`predict` возвращает DataFrame ``[user_id, item_id, score]``;
      * детерминированность при фиксированном ``seed``.
    """

    name: str = "base"
    requires: frozenset[str] = frozenset({"donor_scores"})

    def __init__(self) -> None:
        self._fitted = False

    # --- контракт ---

    @abstractmethod
    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        """Реализация обучения метода (заполняется подклассом)."""

    @abstractmethod
    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        """Предсказать скоры для всех пар (user_ids × cold_item_ids).

        Возвращает long-format DataFrame ``[user_id, item_id, score]``.
        """

    # --- публичный API с валидацией входов ---

    def fit(self, inputs: TransferInputs, seed: int = 0) -> ColdStartMethod:
        """Проверить обязательные входы и обучить метод."""
        self._validate_inputs(inputs)
        self._fit(inputs, seed)
        self._fitted = True
        return self

    def _validate_inputs(self, inputs: TransferInputs) -> None:
        unknown = self.requires - INPUT_KINDS
        if unknown:
            raise ValueError(f"{self.name}: неизвестные требования {unknown}")
        present = _present_inputs(inputs)
        missing = self.requires - present
        if missing:
            raise MissingInputError(
                f"Метод {self.name!r} требует {sorted(self.requires)}, "
                f"но отсутствуют: {sorted(missing)}"
            )

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError(f"{self.name}: вызовите fit() до predict()")

    def get_params(self) -> dict:
        """Гиперпараметры метода (для логирования в артефакты)."""
        return {}


def _present_inputs(inputs: TransferInputs) -> set[str]:
    present: set[str] = set()
    if inputs.donor_scores is not None and len(inputs.donor_scores):
        present.add("donor_scores")
    if inputs.train_interactions is not None and len(inputs.train_interactions):
        present.add("train_interactions")
    if inputs.warm_features is not None or inputs.cold_features is not None:
        present.add("content")
    if inputs.similarity is not None:
        present.add("similarity")
    if inputs.embeddings is not None:
        present.add("embeddings")
    if inputs.item_meta is not None:
        present.add("item_meta")
    if inputs.val_interactions is not None and inputs.val_cold_features is not None:
        present.add("val")
    return present


def cross_join_frame(
    user_ids: np.ndarray, cold_item_ids: np.ndarray, scores: np.ndarray
) -> pd.DataFrame:
    """Утилита: собрать long-format DataFrame из плотной матрицы скоров [n_users, n_items]."""
    n_u, n_i = len(user_ids), len(cold_item_ids)
    if scores.shape != (n_u, n_i):
        raise ValueError(f"scores.shape={scores.shape} != ({n_u}, {n_i})")
    return pd.DataFrame(
        {
            C.User: np.repeat(user_ids, n_i),
            C.Item: np.tile(cold_item_ids, n_u),
            C.Score: scores.reshape(-1),
        }
    )


#: Глобальный реестр методов.
methods: Registry[type[ColdStartMethod]] = Registry("method")


_M = TypeVar("_M", bound=ColdStartMethod)


def register_method(name: str) -> Callable[[type[_M]], type[_M]]:
    """Декоратор регистрации метода под именем ``name`` (проставляет ``cls.name``).

    Сохраняет конкретный тип класса (не схлопывает к базовому), чтобы pyright видел
    сигнатуру ``__init__`` подкласса.
    """

    def decorator(cls: type[_M]) -> type[_M]:
        cls.name = name
        methods.register(name)(cls)
        return cls

    return decorator
