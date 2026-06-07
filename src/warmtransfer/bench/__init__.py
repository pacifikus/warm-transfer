"""warmtransfer.bench — benchmark built on top of warmtransfer.

The heavy part (datasets, donors, runner). Installed via the ``bench`` extra:
``uv sync --extra bench``. The ``warmtransfer`` library itself does not depend on it.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
