"""Tests for user-facing examples."""

from __future__ import annotations

import runpy


def test_quickstart_example_runs() -> None:
    ns = runpy.run_path("examples/quickstart.py")

    reco = ns["reco"]
    assert len(reco) == 2
