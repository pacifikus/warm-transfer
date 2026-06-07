"""End-to-end tests for recommend()."""

from __future__ import annotations

import numpy as np
import pandas as pd

from warmtransfer.columns import Columns as C
from warmtransfer.recommend import AutoResult, recommend


def _synthetic(n_users: int = 60, n_items: int = 40, n_feat: int = 6, seed: int = 0):
    """Content drives affinity -> transfer methods should beat popularity."""
    from warmtransfer.types import ItemFeatures

    rng = np.random.default_rng(seed)
    item_feat = rng.random((n_items, n_feat))
    user_pref = rng.random((n_users, n_feat))
    affinity = user_pref @ item_feat.T  # [n_users, n_items]

    rows = []
    scores = []
    for u in range(n_users):
        top = np.argsort(-affinity[u])[:8]
        for it in top:
            rows.append((u, int(it)))
        for it in range(n_items):
            scores.append((u, it, float(affinity[u, it])))
    inter = pd.DataFrame(rows, columns=[C.User, C.Item])  # type: ignore[arg-type]
    donor = pd.DataFrame(scores, columns=[C.User, C.Item, C.Score])  # type: ignore[arg-type]
    content = ItemFeatures(
        item_ids=np.arange(n_items), matrix=item_feat,
        feature_names=[f"f{i}" for i in range(n_feat)],
    )
    return inter, content, donor


def test_recommend_returns_leaderboard_and_predicts() -> None:
    inter, content, donor = _synthetic()
    res = recommend(inter, content, donor, seed=42, verbose=False)
    assert isinstance(res, AutoResult)
    assert res.metric in res.leaderboard.columns
    assert "linmap" in res.leaderboard.index
    assert res.best in res.leaderboard.index
    cold = np.array([0])
    reco = res.predict(user_ids=np.array([0, 1, 2]), cold_item_ids=cold)
    assert list(reco.columns[:3]) == [C.User, C.Item, C.Score]
    assert set(reco[C.Item]) == {0}


def test_recommend_verdict_mentions_metric() -> None:
    inter, content, donor = _synthetic()
    res = recommend(inter, content, donor, seed=42, verbose=False)
    assert res.metric in res.verdict or "auc" in res.verdict.lower()
    assert isinstance(str(res), str) and len(str(res)) > 0


def test_recommend_skips_embeddings() -> None:
    inter, content, donor = _synthetic()
    res = recommend(inter, content, donor, seed=42, verbose=False)
    assert "dropoutnet" in res.skipped


def test_recommend_empty_interactions_raises() -> None:
    import pytest
    _, content, donor = _synthetic()
    with pytest.raises(ValueError):
        recommend(pd.DataFrame(columns=[C.User, C.Item]), content, donor, verbose=False)  # type: ignore[arg-type]


def test_failing_method_lands_in_skipped(monkeypatch) -> None:
    inter, content, donor = _synthetic()
    from warmtransfer.methods.linmap import LinMap

    def boom(self, inputs, seed):
        raise RuntimeError("boom")

    monkeypatch.setattr(LinMap, "_fit", boom)
    res = recommend(
        inter, content, donor, methods=["linmap", "knn_score_avg"], seed=42, verbose=False
    )
    assert "linmap" in res.skipped
    assert "knn_score_avg" in res.leaderboard.index


def test_predict_different_cold_sets() -> None:
    inter, content, donor = _synthetic()
    res = recommend(inter, content, donor, seed=42, verbose=False)
    r1 = res.predict(np.array([0, 1]), np.array([0]))
    r2 = res.predict(np.array([0, 1]), np.array([2, 3]))
    assert set(r1[C.Item]) == {0}
    assert set(r2[C.Item]) == {2, 3}


def test_public_export() -> None:
    import warmtransfer as wt

    assert hasattr(wt, "recommend")
    assert hasattr(wt, "AutoResult")
    assert hasattr(wt, "HoldoutConfig")


def test_example_runs() -> None:
    import runpy
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "examples" / "recommend_quickstart.py"
    runpy.run_path(str(path), run_name="__main__")
