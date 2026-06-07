"""Benchmark run configuration (pydantic — validation at the boundary)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ComponentCfg(BaseModel):
    """A component referenced by its registry name plus its parameters.

    :param label: name used in the results table (defaults to ``name``). Needed for ablations —
        one method with different hyperparameters under different labels in a single run.
    """

    name: str
    params: dict = Field(default_factory=dict)
    label: str | None = None

    @property
    def key(self) -> str:
        """Label used for grouping in the results."""
        return self.label or self.name


class SplitterCfg(BaseModel):
    name: str = "pseudo_cold"
    params: dict = Field(default_factory=dict)


class BenchConfig(BaseModel):
    """Run specification: datasets × donors × methods × seeds."""

    datasets: list[str]
    donors: list[ComponentCfg]
    methods: list[ComponentCfg]
    splitter: SplitterCfg = Field(default_factory=SplitterCfg)
    metrics_ks: tuple[int, ...] = (1, 5, 10)
    seeds: list[int] = Field(default_factory=lambda: [42])
    max_eval_users: int | None = 2000
    out_dir: str = "benchmarks_out"

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
