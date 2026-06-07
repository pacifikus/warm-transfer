"""Collecting and exporting benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from warmtransfer.metrics.relative import rela_impr


def to_table(records: list[dict], *, with_std: bool = False) -> pd.DataFrame:
    """List of run records -> DataFrame, averaged over seeds.

    :param with_std: add ``<metric>_std`` columns — the standard deviation over seeds
        (for assessing stability across multiple seeds).
    """
    df = pd.DataFrame(records)
    group_cols = ["dataset", "donor", "method"]
    metric_cols = [c for c in df.columns if c not in {*group_cols, "seed"}]
    grouped = df.groupby(group_cols, as_index=False)
    agg = cast("pd.DataFrame", grouped[metric_cols].mean())
    if with_std:
        std = cast("pd.DataFrame", grouped[metric_cols].std(ddof=0))
        for c in metric_cols:
            agg[f"{c}_std"] = cast("pd.Series", std[c]).to_numpy()
    return agg


def add_rela_impr(
    table: pd.DataFrame, base_method: str, metric: str = "auc"
) -> pd.DataFrame:
    """Add a ``rela_impr`` column — the relative AUC improvement over ``base_method``.

    RelaImpr = (AUC_model − 0.5)/(AUC_base − 0.5) − 1, computed within each
    (dataset, donor) pair. For the base method itself = 0.0; if the base is missing — NaN.
    """
    out = table.copy()
    rela: list[float] = []
    for _, row in out.iterrows():
        mask = (
            (out["dataset"] == row["dataset"])
            & (out["donor"] == row["donor"])
            & (out["method"] == base_method)
        )
        base_rows = out.loc[mask, metric]
        if len(base_rows) == 0:
            rela.append(float("nan"))
        else:
            rela.append(rela_impr(float(row[metric]), float(base_rows.to_numpy()[0])))
    out["rela_impr"] = rela
    return out


def save_table(df: pd.DataFrame, out_dir: str | Path, name: str = "results") -> dict[str, Path]:
    """Save the table to parquet + markdown. Returns the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    parquet = out / f"{name}.parquet"
    df.to_parquet(parquet, index=False)
    paths["parquet"] = parquet
    md = out / f"{name}.md"
    md.write_text(df.to_markdown(index=False, floatfmt=".4f") or "", encoding="utf-8")
    paths["markdown"] = md
    return paths
