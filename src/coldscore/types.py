"""Внутренние типы данных (горячий путь): dataclasses + numpy/pandas, без pydantic.

Pydantic применяется только на границах (конфиги, валидация публичного входа в
``schema.py``). Здесь — лёгкие контейнеры для обмена между компонентами.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

#: Матрица признаков/сходства: плотная или разреженная.
Matrix = np.ndarray | sparse.spmatrix


@dataclass
class ItemFeatures:
    """Контентные признаки айтемов, выровненные по ``item_ids``.

    ``matrix[i]`` — вектор признаков айтема ``item_ids[i]`` (внешний id).
    """

    item_ids: np.ndarray  # внешние id, shape [n_items]
    matrix: Matrix  # [n_items, n_features]
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.item_ids)
        if self.matrix.shape[0] != n:
            raise ValueError(f"matrix.shape[0]={self.matrix.shape[0]} != len(item_ids)={n}")
        # карта внешний id -> позиция строки
        self._pos: dict = {iid: i for i, iid in enumerate(self.item_ids)}

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_features(self) -> int:
        return int(self.matrix.shape[1])

    def subset(self, ids: np.ndarray) -> ItemFeatures:
        """Подвыборка по внешним id (порядок — как в ``ids``)."""
        rows = [self._pos[i] for i in ids]
        return ItemFeatures(
            item_ids=np.asarray(ids),
            matrix=self.matrix[rows],  # type: ignore[index]  # np.ndarray и sparse оба поддерживают
            feature_names=self.feature_names,
        )

    def has(self, item_id: object) -> bool:
        return item_id in self._pos


@dataclass
class Dataset:
    """Датасет: взаимодействия + контент айтемов.

    Взаимодействия — long-format DataFrame с колонками ``Columns`` (User, Item,
    Weight, Datetime). ``item_features`` опционально (контент может делегироваться
    пользователю библиотеки).
    """

    interactions: pd.DataFrame
    item_features: ItemFeatures | None = None
    name: str = "dataset"

    @property
    def all_items(self) -> np.ndarray:
        from coldscore.columns import Columns as C

        return np.asarray(self.interactions[C.Item].unique())


@dataclass
class SplitResult:
    """Результат честного cold-start сплита.

    Инвариант (проверяется тестами): ``cold_items`` не встречаются ни в ``train``,
    ни в ``val``; их взаимодействия присутствуют только в ``test``.
    """

    train: pd.DataFrame  # warm-взаимодействия для обучения донора
    val: pd.DataFrame  # warm/cold-валидация (для тюнинга гиперпараметров)
    test: pd.DataFrame  # взаимодействия cold-айтемов (ground truth)
    warm_items: np.ndarray
    cold_items: np.ndarray

    def cold_in_train(self) -> set:
        """Множество cold-айтемов, протёкших в train (должно быть пустым)."""
        from coldscore.columns import Columns as C

        return set(self.cold_items) & set(self.train[C.Item].unique())


@dataclass
class TransferInputs:
    """Всё, что cold-start метод получает на вход (``ColdStartMethod.fit``).

    Бандл собирается раннером бенчмарка из ``Dataset``/``SplitResult`` и донора.
    При прямом использовании библиотеки пользователь заполняет нужные поля сам
    (минимум — ``donor_scores`` + один из источников контента/сходства).
    """

    donor_scores: pd.DataFrame  # [User, Item, Score] по WARM-айтемам
    train_interactions: pd.DataFrame = field(
        default_factory=pd.DataFrame
    )  # для popularity-бейзлайнов (Grouped MP)
    warm_features: ItemFeatures | None = None
    cold_features: ItemFeatures | None = None
    # similarity: [n_cold, n_warm], выровнено по *_features.item_ids
    similarity: np.ndarray | None = None
    # embeddings: {"warm_item": ..., "cold_item": ..., "user": ...}
    embeddings: dict[str, np.ndarray] | None = None
    warm_items: np.ndarray | None = None
    cold_items: np.ndarray | None = None
    item_meta: pd.DataFrame | None = None  # доп. мета (категория айтема) для Grouped MP

    # --- val-cold фолд (для супервизорных методов: stacking, logreg-калибровка, тюнинг) ---
    # Эти айтемы тоже cold (нет в train донора), но их взаимодействия известны (split.val),
    # поэтому на них можно обучать мета-модель и предсказывать уже на тестовых cold.
    val_cold_features: ItemFeatures | None = None
    val_similarity: np.ndarray | None = None  # [n_val_cold, n_warm]
    val_interactions: pd.DataFrame | None = None  # ground truth для val-cold
