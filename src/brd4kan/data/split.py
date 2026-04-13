"""Stage 3 — Bemis-Murcko scaffold split + first-publication-year time split.

The scaffold split is the canonical DeepChem-style group split: every
Bemis-Murcko scaffold lives in exactly one of train / val / test, so the
test ``test_split.py`` can assert *zero* shared scaffolds across sets.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from brd4kan.utils.config import Params
from brd4kan.utils.hashing import file_signature
from brd4kan.utils.manifest import (
    Manifest,
    env_snapshot,
    get_git_sha,
    utc_timestamp,
    write_manifest,
)
from brd4kan.utils.runs import make_run_dir

logger = logging.getLogger(__name__)


def _import_rdkit() -> Any:
    from rdkit import Chem

    return Chem


def bemis_murcko_scaffold(smiles: str, include_chirality: bool = False) -> str | None:
    """Return the Bemis-Murcko scaffold SMILES, or ``None`` if RDKit cannot parse."""
    Chem = _import_rdkit()
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def scaffold_split(
    smiles: list[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    include_chirality: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    """Group-by-scaffold split. Largest scaffolds go to train first, then val, then test.

    The result is fully deterministic in ``smiles`` order — no RNG involved.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(
            f"split fractions must sum to 1.0, got {train_frac + val_frac + test_frac}"
        )
    if train_frac < 0 or val_frac < 0 or test_frac < 0:
        raise ValueError("split fractions must be non-negative")

    n = len(smiles)
    if n == 0:
        return [], [], []

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles):
        s = bemis_murcko_scaffold(smi, include_chirality=include_chirality)
        # Unparseable / acyclic compounds get a unique singleton scaffold so
        # they can never share a group with anything else (no leakage risk).
        if not s:
            s = f"__singleton_{idx}__"
        groups[s].append(idx)

    sorted_groups = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    train_target = int(round(train_frac * n))
    val_target = int(round(val_frac * n))

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for _, idxs in sorted_groups:
        if len(train) + len(idxs) <= train_target:
            train.extend(idxs)
        elif len(val) + len(idxs) <= val_target:
            val.extend(idxs)
        else:
            test.extend(idxs)
    logger.info(
        "scaffold split: train=%d val=%d test=%d (n_scaffolds=%d)",
        len(train),
        len(val),
        len(test),
        len(groups),
    )
    return train, val, test


def time_split(
    df: pd.DataFrame, year_field: str, test_year_quantile: float
) -> tuple[list[int], list[int]]:
    """Time split: rows below the year quantile → train, ≥ → test."""
    if year_field not in df.columns:
        raise KeyError(f"time_split requires column '{year_field}'")
    years = df[year_field].dropna()
    if years.empty:
        return df.index.tolist(), []
    cutoff = float(np.quantile(years.to_numpy(), test_year_quantile))
    train_idx = df.index[df[year_field] < cutoff].tolist()
    test_idx = df.index[df[year_field] >= cutoff].tolist()
    logger.info(
        "time split: cutoff_year=%.2f train=%d test=%d", cutoff, len(train_idx), len(test_idx)
    )
    return train_idx, test_idx


def save_split_files(
    out_dir: Path,
    scaffold_indices: tuple[list[int], list[int], list[int]],
    time_indices: tuple[list[int], list[int]],
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = scaffold_indices
    scaffold_path = out_dir / "scaffold_split.json"
    scaffold_path.write_text(
        json.dumps({"train": train, "val": val, "test": test}, indent=2),
        encoding="utf-8",
    )
    t_train, t_test = time_indices
    time_path = out_dir / "time_split.json"
    time_path.write_text(
        json.dumps({"train": t_train, "test": t_test}, indent=2),
        encoding="utf-8",
    )
    return {"scaffold": scaffold_path, "time": time_path}


def run_split(in_path: Path, out_dir: Path, params: Params) -> dict[str, Path]:
    """End-to-end Stage 3: curated parquet → scaffold + time split JSONs + manifest."""
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    smiles = df["canonical_smiles_std"].tolist()
    scaffold_indices = scaffold_split(
        smiles,
        params.split.scaffold.train_frac,
        params.split.scaffold.val_frac,
        params.split.scaffold.test_frac,
        include_chirality=params.split.scaffold.include_chirality,
    )
    time_indices = time_split(
        df, params.split.time.year_field, params.split.time.test_year_quantile
    )
    paths = save_split_files(out_dir, scaffold_indices, time_indices)

    run_dir = make_run_dir(out_dir)
    manifest = Manifest(
        stage="split",
        git_sha=get_git_sha(),
        timestamp=utc_timestamp(),
        inputs={"curated_parquet": file_signature(in_path)},
        outputs={
            "scaffold_split_json": file_signature(paths["scaffold"]),
            "time_split_json": file_signature(paths["time"]),
            "n_train_scaffold": len(scaffold_indices[0]),
            "n_val_scaffold": len(scaffold_indices[1]),
            "n_test_scaffold": len(scaffold_indices[2]),
            "n_train_time": len(time_indices[0]),
            "n_test_time": len(time_indices[1]),
        },
        params_snapshot=params.model_dump(),
        seeds={"global": params.seed},
        env=env_snapshot(),
        wall_time_seconds=round(time.perf_counter() - started, 4),
    )
    write_manifest(manifest, run_dir)
    return paths
