"""Scale & Shift — облегчённый MWUF на уровне скоров (score-only).

MWUF (Zhu et al., SIGIR 2021) переводит cold ID-эмбеддинг в «тёплое» пространство двумя
сетями: scaling (масштаб) и shifting (сдвиг). Здесь переносим саму идею на СКОРЫ, без
доступа к эмбеддингам и без обучения сетей.

Базовый сырой скор cold-айтема собираем как у наивного KNN — среднее скоров донора по
k контентным warm-соседям. Затем для каждого cold-айтема стандартизуем его вектор скоров
по пользователям и масштабируем к типичной warm-статистике (scale ``sigma*``, shift ``mu*``):

    raw[u, c]  = Σ_j w_{cj} · donor[u, j]        (контентный KNN)
    out[u, c]  = (raw[u, c] - m_c) / (s_c + eps) · sigma* + mu*,

где ``m_c, s_c`` — среднее/стд сырого скора cold-айтема по пользователям, а ``mu*, sigma*`` —
средние по warm-айтемам среднего/стд их скоров. Эффект: убирается раздутый общий уровень
скора популярных cold-айтемов (дебиасинг), а распределение подгоняется под warm. Нужны только
скоры донора + контент (для соседей), эмбеддинги не нужны ([MA], score-only).

Ссылка (идея scale/shift): Zhu et al., "Learning to Warm Up Cold Item Embeddings for
Cold-start Recommendation with Meta Scaling and Shifting Networks", SIGIR 2021.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.methods.base import ColdStartMethod, cross_join_frame, register_method
from warmtransfer.methods.debiased_knn import _pivot_scores, _topk_neighbors
from warmtransfer.types import TransferInputs

_EPS = 1e-8


@register_method("scale_shift")
class ScaleShift(ColdStartMethod):
    """Контентный KNN над скорами донора + калибровка к warm-статистикам.

    :param k: число ближайших warm-соседей.
    :param clip_negative: обнулять отрицательные сходства.
    """

    requires = frozenset({"donor_scores", "similarity", "content"})

    def __init__(self, k: int = 20, clip_negative: bool = True) -> None:
        super().__init__()
        self.k = k
        self.clip_negative = clip_negative

    def _fit(self, inputs: TransferInputs, seed: int) -> None:
        if inputs.warm_features is None or inputs.cold_features is None:
            raise ValueError("scale_shift требует warm_features и cold_features")
        if inputs.similarity is None:
            raise ValueError("scale_shift требует similarity [n_cold, n_warm]")

        self._warm_ids = np.asarray(inputs.warm_features.item_ids)
        self._cold_ids = np.asarray(inputs.cold_features.item_ids)
        self._cold_pos = {it: r for r, it in enumerate(self._cold_ids)}

        # матрица скоров донора [n_users, n_warm], выровнена по self._warm_ids
        donor, self._user_ids = _pivot_scores(inputs.donor_scores, self._warm_ids)
        self._user_pos = {u: i for i, u in enumerate(self._user_ids)}

        # целевая warm-статистика: средние по айтемам среднего/стд скора по пользователям
        self._mu_star = float(donor.mean(axis=0).mean())
        self._sigma_star = float(donor.std(axis=0).mean())

        sim = np.asarray(inputs.similarity, dtype=float)
        if self.clip_negative:
            sim = np.clip(sim, 0.0, None)
        neighbors, weights = _topk_neighbors(sim, self.k)

        # сырой скор cold-айтемов [n_users, n_cold] = KNN-усреднение скоров донора
        raw = np.zeros((donor.shape[0], len(self._cold_ids)))
        for r in range(len(self._cold_ids)):
            raw[:, r] = donor[:, neighbors[r]] @ weights[r]

        # стандартизация по пользователям + scale&shift к warm-статистике
        m_c = raw.mean(axis=0)  # [n_cold]
        s_c = raw.std(axis=0)  # [n_cold]
        z = (raw - m_c) / (s_c + _EPS)
        self._cold_scores = z * self._sigma_star + self._mu_star  # [n_users, n_cold]

    def predict(self, user_ids: np.ndarray, cold_item_ids: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        user_ids = np.asarray(user_ids)
        cold_item_ids = np.asarray(cold_item_ids)

        rows = np.array([self._user_pos.get(u, -1) for u in user_ids])
        cols = np.array([self._cold_pos.get(it, -1) for it in cold_item_ids])

        scores = np.zeros((len(user_ids), len(cold_item_ids)))
        u_known = rows >= 0
        for c, col in enumerate(cols):
            if col < 0:
                continue
            # неизвестный пользователь → mu* (нейтральный warm-уровень)
            scores[~u_known, c] = self._mu_star
            scores[u_known, c] = self._cold_scores[rows[u_known], col]
        return cross_join_frame(user_ids, cold_item_ids, scores)

    def get_params(self) -> dict:
        return {"k": self.k, "clip_negative": self.clip_negative}
