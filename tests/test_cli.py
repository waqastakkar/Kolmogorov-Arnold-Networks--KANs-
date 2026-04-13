"""Smoke tests for the ``brd4kan`` Typer CLI (stages 1-4)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

rdkit = pytest.importorskip("rdkit")
csp = pytest.importorskip("chembl_structure_pipeline")
mordred = pytest.importorskip("mordred")

from brd4kan.cli import app  # noqa: E402

runner = CliRunner()


def test_help_lists_stage_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout
    assert "extract" in out
    assert "curate" in out
    assert "split" in out
    assert "featurize" in out


def test_extract_requires_db_when_env_unset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CHEMBL_DB_PATH", raising=False)
    result = runner.invoke(app, ["extract", "--out", str(tmp_path / "raw")])
    assert result.exit_code != 0


def test_extract_runs_with_explicit_db(tmp_path: Path, tiny_chembl_db: Path) -> None:
    out_dir = tmp_path / "raw"
    result = runner.invoke(
        app,
        ["extract", "--out", str(out_dir), "--db", str(tiny_chembl_db)],
    )
    assert result.exit_code == 0, result.stdout
    assert (out_dir / "brd4_raw.parquet").exists()
    # The CLI prints the canonical output path on success.
    assert "brd4_raw.parquet" in result.stdout
