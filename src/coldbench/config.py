"""Конфигурация прогона бенчмарка (pydantic — валидация на границе)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ComponentCfg(BaseModel):
    """Компонент по имени из реестра + его параметры.

    :param label: имя в таблице результатов (по умолчанию ``name``). Нужен для абляций —
        один метод с разными гиперпараметрами под разными метками в одном прогоне.
    """

    name: str
    params: dict = Field(default_factory=dict)
    label: str | None = None

    @property
    def key(self) -> str:
        """Метка для группировки в результатах."""
        return self.label or self.name


class SplitterCfg(BaseModel):
    name: str = "pseudo_cold"
    params: dict = Field(default_factory=dict)


class BenchConfig(BaseModel):
    """Описание прогона: датасеты × доноры × методы × сиды."""

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
