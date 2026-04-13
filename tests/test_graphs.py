"""Trivial covers for the Chemprop CSV writer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from brd4kan.features.graphs import save_chemprop_csv


def test_save_chemprop_csv_round_trip(tmp_path: Path) -> None:
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    targets = [1.0, 2.0, 3.0]
    out = save_chemprop_csv(smiles, targets, tmp_path / "x.csv")
    assert out.exists()
    df = pd.read_csv(out)
    assert df["smiles"].tolist() == smiles
    assert df["pchembl_value"].tolist() == targets


def test_save_chemprop_csv_length_mismatch_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_chemprop_csv(["CCO"], [1.0, 2.0], tmp_path / "y.csv")
