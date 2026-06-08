"""Smoke test for the `warmbench try` subcommand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from warmtransfer.bench.__main__ import main
from warmtransfer.columns import Columns as C

pytestmark = pytest.mark.bench


def _write_inputs(tmp_path):
    rng = np.random.default_rng(0)
    n_users, n_items, n_feat = 50, 30, 5
    item_feat = rng.random((n_items, n_feat))
    user_pref = rng.random((n_users, n_feat))
    affinity = user_pref @ item_feat.T
    rows, scores = [], []
    for u in range(n_users):
        for it in np.argsort(-affinity[u])[:6]:
            rows.append((u, int(it)))
        for it in range(n_items):
            scores.append((u, it, float(affinity[u, it])))
    inter = pd.DataFrame(rows, columns=[C.User, C.Item])  # type: ignore[call-overload]
    donor = pd.DataFrame(scores, columns=[C.User, C.Item, C.Score])  # type: ignore[call-overload]
    content = pd.DataFrame(item_feat, columns=[f"f{i}" for i in range(n_feat)])  # type: ignore[call-overload]
    content.insert(0, C.Item, np.arange(n_items))

    ip, dp, cp = tmp_path / "i.parquet", tmp_path / "d.parquet", tmp_path / "c.parquet"
    inter.to_parquet(ip)
    donor.to_parquet(dp)
    content.to_parquet(cp)
    return ip, dp, cp


def test_try_runs(tmp_path, capsys):
    ip, dp, cp = _write_inputs(tmp_path)
    rc = main(["try", "--interactions", str(ip), "--scores", str(dp), "--content", str(cp)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Verdict" in out


def test_try_writes_out_file(tmp_path, capsys):
    ip, dp, cp = _write_inputs(tmp_path)
    out = tmp_path / "report.md"
    rc = main(["try", "--interactions", str(ip), "--scores", str(dp),
               "--content", str(cp), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert "Verdict" in out.read_text()
