"""Воспроизводимость: единая точка фиксации сидов.

``torch`` фиксируется опционально (он в extra ``deep``), чтобы core не зависел от него.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Зафиксировать сиды ``random``, ``numpy`` и (если установлен) ``torch``.

    Вызывается в начале каждого ``fit`` метода/адаптера и в начале прогона бенчмарка.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_rng(seed: int) -> np.random.Generator:
    """Локальный генератор numpy (предпочтительнее глобального состояния)."""
    return np.random.default_rng(seed)
