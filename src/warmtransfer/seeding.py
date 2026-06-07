"""Reproducibility: a single point for seeding.

``torch`` is seeded optionally (it lives in the ``deep`` extra), so that core does not depend on it.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed ``random``, ``numpy`` and (if installed) ``torch``.

    Called at the start of every method/adapter ``fit`` and at the start of a benchmark run.
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
    """Local numpy generator (preferable to global state)."""
    return np.random.default_rng(seed)
