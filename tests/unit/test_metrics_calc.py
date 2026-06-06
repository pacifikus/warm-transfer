"""Тесты сборки метрик поверх DataFrame: calc_metrics, mean_user_auc."""

from __future__ import annotations

import pandas as pd
import pytest

from coldscore.columns import Columns as C
from coldscore.metrics import calc_metrics, mean_user_auc, ranking_metrics


def _grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    # 2 пользователя, кандидаты-айтемы {100,101,102} (полная сетка)
    reco = pd.DataFrame(
        {
            C.User: [1, 1, 1, 2, 2, 2],
            C.Item: [100, 101, 102, 100, 101, 102],
            C.Score: [0.9, 0.1, 0.5, 0.2, 0.8, 0.3],
        }
    )
    gt = pd.DataFrame({C.User: [1, 2], C.Item: [100, 101]})
    return reco, gt


def test_mean_user_auc_perfect() -> None:
    reco, gt = _grid()
    # оба пользователя: релевантный айтем имеет максимальный скор → AUC=1.0 у каждого
    assert mean_user_auc(reco, gt) == pytest.approx(1.0)


def test_ranking_metrics_perfect_at_1() -> None:
    reco, gt = _grid()
    m = ranking_metrics(reco, gt, ks=(1,))
    assert m["recall@1"] == pytest.approx(1.0)
    assert m["ndcg@1"] == pytest.approx(1.0)
    assert m["precision@1"] == pytest.approx(1.0)


def test_calc_metrics_keys() -> None:
    reco, gt = _grid()
    m = calc_metrics(reco, gt, ks=(1, 5))
    assert "auc" in m
    assert {"recall@1", "recall@5", "ndcg@1", "map@5", "mrr@1"} <= set(m)


def test_users_without_relevant_excluded() -> None:
    # пользователь 3 без релевантных в gt не должен ломать усреднение
    reco = pd.DataFrame(
        {C.User: [1, 1, 3, 3], C.Item: [10, 11, 10, 11], C.Score: [0.9, 0.1, 0.5, 0.5]}
    )
    gt = pd.DataFrame({C.User: [1], C.Item: [10]})
    m = ranking_metrics(reco, gt, ks=(1,))
    assert m["recall@1"] == pytest.approx(1.0)
