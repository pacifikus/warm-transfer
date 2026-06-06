"""coldbench — бенчмарк поверх coldscore.

Тяжёлая часть (датасеты, доноры, runner). Ставится через extra ``bench``:
``uv sync --extra bench``. Сама библиотека ``coldscore`` от него не зависит.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
