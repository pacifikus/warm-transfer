"""Magnitude Scaling — post-hoc дебиасинг популярности по магнитуде эмбеддинга.

Меняет только длину (норму) вектора cold-айтема, не направление. Магнитуда эмбеддинга —
прокси популярности: айтемы с большой нормой получают систематически высокие скоры
(``score = user · item``). Перетягиваем норму каждого cold-айтема к средней норме warm-айтемов
``mu_w``, балансируя распределение предсказаний по айтемам:

    gamma_c = (||x_c|| + alpha * mu_w) / (||x_c|| * (1 + alpha)),
    x_c' = gamma_c * x_c    =>    ||x_c'|| = (||x_c|| + alpha * mu_w) / (1 + alpha),

то есть новая норма — выпуклая комбинация ||x_c|| и mu_w с весом alpha/(1+alpha) на mu_w
(alpha→0: без изменений; alpha→inf: все нормы стянуты к mu_w).

Генератор cold-эмбеддингов берём от ``linmap_emb`` (Gantner): это даёт чистую абляцию
«linmap_emb vs +magnitude scaling» — изолированный эффект дебиасинга. Не требует обучения
самого скейлинга, только статистики warm-эмбеддингов ([EMB]).

Ссылка: Meehan & Pauwels, "On Inherited Popularity Bias in Cold-Start Item
Recommendation", RecSys 2025.
"""

from __future__ import annotations

import numpy as np

from coldscore.methods.base import register_method
from coldscore.methods.linmap_emb import LinMapEmbedding


@register_method("magnitude_scaling")
class MagnitudeScaling(LinMapEmbedding):
    """Gantner-генерация cold-эмбеддингов + дебиасинг популярности по магнитуде.

    :param alpha: L2-регуляризация Ridge (генератор контент → факторы).
    :param ms_alpha: сила стягивания нормы к ``mu_w`` (0 — без дебиасинга).
    """

    def __init__(self, alpha: float = 10.0, ms_alpha: float = 1.0) -> None:
        super().__init__(alpha=alpha)
        self.ms_alpha = ms_alpha

    def _postprocess_cold_emb(self) -> None:
        mu_w = self._warm_mean_norm
        norms = np.linalg.norm(self._cold_emb, axis=1)
        new_norms = (norms + self.ms_alpha * mu_w) / (1.0 + self.ms_alpha)
        # gamma = new_norm / norm; для нулевого вектора оставляем ноль (нет направления)
        gamma = np.divide(
            new_norms, norms, out=np.zeros_like(norms), where=norms > 0
        )
        self._cold_emb = self._cold_emb * gamma[:, None]

    def get_params(self) -> dict:
        return {"alpha": self.alpha, "ms_alpha": self.ms_alpha}
