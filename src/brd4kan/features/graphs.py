"""Graph featurization for the Chemprop D-MPNN baseline.

Chemprop ingests SMILES + targets directly via CSV; we just snapshot a
deterministic, training-ready CSV per split. The actual graph tensorization
happens inside Chemprop at training time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_chemprop_csv(
    smiles: list[str],
    targets: list[float],
    out_path: Path,
    smiles_column: str = "smiles",
    target_column: str = "pchembl_value",
) -> Path:
    """Write a Chemprop-compatible CSV with stable column order."""
    if len(smiles) != len(targets):
        raise ValueError("smiles and targets must have the same length")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({smiles_column: smiles, target_column: targets})
    df.to_csv(out_path, index=False)
    return out_path
