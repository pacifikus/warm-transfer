"""Attention-взвешенное усреднение эмбеддингов контентных соседей (SimCSR на уровне IT).

Расширение ``embedding_avg``: вместо равномерного среднего эмбеддингов k ближайших по
контенту warm-соседей берём softmax-взвешенное по контентному сходству. Это «полный» SimCSR
в пространстве латентных факторов донора: query = контент cold-айтема, key = контент соседа
(через similarity), value = эмбеддинг соседа. Скор = user_emb · cold_emb. Нужны эмбеддинги ([EMB]).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.methods.base import ColdStartMethod, cross_join_frame, register_method
from warmtransfer.types import TransferInputs


@register_method("attention_emb")
class AttentionEmbedding(ColdStartMethod):
    """Softmax-взвешенное по контенту усреднение эмбеддингов соседей донора.

    :param k: число ближайших warm-соседей.
    :param temperature: температура softmax (меньше → острее веса).
    """

    requires = frozenset({"embeddings", "similarity", "content"})

    def __init__(self, k: int = 20, temperature: float = 0.1) -> None:
        super().__init__()
        self.k = k
        self.temperature = temperature

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        if inputs.warm_features is None or inputs.cold_features is None:
            raise ValueError("attention_emb требует warm_features и cold_features")
        if inputs.similarity is None:
            raise ValueError("attention_emb требует similarity [n_cold, n_warm]")
        if inputs.embeddings is None:
            raise ValueError("attention_emb требует embeddings")
        emb = inputs.embeddings
        for key in ("item", "item_ids", "user", "user_ids"):
            if key not in emb:
                raise ValueError(f"embeddings: отсутствует ключ {key!r}")

        item_emb = np.asarray(emb["item"], dtype=float)
        item_emb_ids = np.asarray(emb["item_ids"])
        self._user_emb = np.asarray(emb["user"], dtype=float)
        user_ids = np.asarray(emb["user_ids"])
        self._user_pos = {u: i for i, u in enumerate(user_ids)}

        warm_ids = np.asarray(inputs.warm_features.item_ids)
        self._cold_ids = np.asarray(inputs.cold_features.item_ids)
        self._cold_pos = {it: r for r, it in enumerate(self._cold_ids)}

        emb_pos = {it: j for j, it in enumerate(item_emb_ids)}
        warm_to_emb = np.array([emb_pos.get(it, -1) for it in warm_ids])

        sim = np.asarray(inputs.similarity, dtype=float)  # [n_cold, n_warm]
        kk = min(self.k, sim.shape[1])
        d = item_emb.shape[1]

        self._cold_emb = np.zeros((len(self._cold_ids), d))
        for r, row in enumerate(sim):
            idx = np.argpartition(-row, kk - 1)[:kk]
            emb_rows = warm_to_emb[idx]
            valid = emb_rows >= 0
            if not valid.any():
                continue
            w = _softmax(row[idx][valid] / self.temperature)
            self._cold_emb[r] = w @ item_emb[emb_rows[valid]]

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        user_ids = np.asarray(user_ids)
        cold_item_ids = np.asarray(cold_item_ids)

        rows = np.array([self._user_pos.get(u, -1) for u in user_ids])
        known = rows >= 0
        u_emb = np.zeros((len(user_ids), self._user_emb.shape[1]))
        u_emb[known] = self._user_emb[rows[known]]

        c_rows = np.array([self._cold_pos.get(it, -1) for it in cold_item_ids])
        c_known = c_rows >= 0
        c_emb = np.zeros((len(cold_item_ids), self._cold_emb.shape[1]))
        c_emb[c_known] = self._cold_emb[c_rows[c_known]]

        scores = u_emb @ c_emb.T
        return cross_join_frame(user_ids, cold_item_ids, scores)

    def get_params(self) -> dict:
        return {"k": self.k, "temperature": self.temperature}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
