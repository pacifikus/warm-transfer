"""Наивное усреднение эмбеддингов контентных соседей.

Эмбеддинг cold-айтема строится как среднее эмбеддингов его k ближайших по контенту
warm-соседей; скор пары (user, cold_item) = скалярное произведение эмбеддинга
пользователя на эмбеддинг cold-айтема. Идея: перенести cold-айтем в латентное
пространство донора через известные warm-соседей и оценивать через user-эмбеддинги.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.methods.base import ColdStartMethod, cross_join_frame, register_method
from warmtransfer.types import TransferInputs


@register_method("embedding_avg")
class EmbeddingAverage(ColdStartMethod):
    """Скоры через усреднённые эмбеддинги контентных warm-соседей.

    :param k: число ближайших warm-соседей для усреднения эмбеддингов.
    """

    requires = frozenset({"embeddings", "similarity", "content"})

    def __init__(self, k: int = 20) -> None:
        super().__init__()
        self.k = k

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        if inputs.warm_features is None or inputs.cold_features is None:
            raise ValueError("embedding_avg требует warm_features и cold_features")
        if inputs.similarity is None:
            raise ValueError("embedding_avg требует similarity [n_cold, n_warm]")
        if inputs.embeddings is None:
            raise ValueError("embedding_avg требует embeddings")

        emb = inputs.embeddings
        for key in ("item", "item_ids", "user", "user_ids"):
            if key not in emb:
                raise ValueError(f"embeddings: отсутствует ключ {key!r}")

        item_emb = np.asarray(emb["item"], dtype=float)  # [m, d]
        item_emb_ids = np.asarray(emb["item_ids"])  # [m]
        self._user_emb = np.asarray(emb["user"], dtype=float)  # [p, d]
        user_emb_ids = np.asarray(emb["user_ids"])  # [p]
        self._user_pos = {u: i for i, u in enumerate(user_emb_ids)}

        warm_ids = np.asarray(inputs.warm_features.item_ids)  # столбцы similarity
        self._cold_ids = np.asarray(inputs.cold_features.item_ids)  # строки similarity
        self._cold_pos = {it: r for r, it in enumerate(self._cold_ids)}

        # позиция warm-айтема в item_emb по его внешнему id
        emb_pos = {it: j for j, it in enumerate(item_emb_ids)}
        # warm-позиция (столбец similarity) -> строка в item_emb (или -1, если эмбеддинга нет)
        warm_to_emb = np.array([emb_pos.get(it, -1) for it in warm_ids])

        sim = np.asarray(inputs.similarity, dtype=float)  # [n_cold, n_warm]
        kk = min(self.k, sim.shape[1])
        d = item_emb.shape[1]

        # для каждого cold-айтема: среднее эмбеддингов k ближайших warm-соседей
        self._cold_emb = np.zeros((len(self._cold_ids), d))
        for r, row in enumerate(sim):
            idx = np.argpartition(-row, kk - 1)[:kk]
            emb_rows = warm_to_emb[idx]
            emb_rows = emb_rows[emb_rows >= 0]  # соседи без эмбеддинга пропускаем
            if emb_rows.size:
                self._cold_emb[r] = item_emb[emb_rows].mean(axis=0)

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        user_ids = np.asarray(user_ids)
        cold_item_ids = np.asarray(cold_item_ids)

        # эмбеддинги запрошенных пользователей (неизвестный юзер -> нулевой вектор)
        rows = np.array([self._user_pos.get(u, -1) for u in user_ids])
        known = rows >= 0
        u_emb = np.zeros((len(user_ids), self._user_emb.shape[1]))
        u_emb[known] = self._user_emb[rows[known]]

        # эмбеддинги запрошенных cold-айтемов (неизвестный айтем -> нулевой вектор)
        c_rows = np.array([self._cold_pos.get(it, -1) for it in cold_item_ids])
        c_known = c_rows >= 0
        c_emb = np.zeros((len(cold_item_ids), self._cold_emb.shape[1]))
        c_emb[c_known] = self._cold_emb[c_rows[c_known]]

        scores = u_emb @ c_emb.T  # [n_users, n_cold]
        return cross_join_frame(user_ids, cold_item_ids, scores)

    def get_params(self) -> dict:
        return {"k": self.k}
