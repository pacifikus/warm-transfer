"""Тесты CLI coldbench без запуска тяжёлого бенчмарка."""

from __future__ import annotations

from pathlib import Path

from coldbench.__main__ import main


def test_cli_list_components(capsys) -> None:
    code = main(["--list-components"])

    out = capsys.readouterr().out
    assert code == 0
    assert "datasets:" in out
    assert "donors:" in out
    assert "methods:" in out
    assert "ml-1m" in out
    assert "als" in out
    assert "linmap" in out


def test_cli_dry_run_validates_config(tmp_path: Path, capsys) -> None:
    config = tmp_path / "bench.yaml"
    config.write_text(
        """
datasets: [ml-1m]
donors:
  - name: als
methods:
  - name: linmap
seeds: [42]
""",
        encoding="utf-8",
    )

    code = main(["--config", str(config), "--dry-run"])

    out = capsys.readouterr().out
    assert code == 0
    assert "dry-run ok" in out
    assert "datasets: ml-1m" in out
    assert "donors: als" in out
    assert "methods: linmap" in out
