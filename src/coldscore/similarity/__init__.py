"""Контентное сходство cold→warm (опционально; может делегироваться пользователю).

Наполняется в Фазе 1+ (cosine, kNN-граф). Здесь — точка сборки.
"""

from __future__ import annotations

from coldscore.similarity.content import content_similarity

__all__ = ["content_similarity"]
