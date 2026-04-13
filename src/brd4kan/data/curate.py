"""Stage 2 — Curate raw activities into a clean QSAR dataset.

Steps (PLAN.md §2 step 2):

1. ChEMBL Structure Pipeline → standardize / desalt / neutralize → parent.
2. Reject MW ∉ [mw_min, mw_max], heavy atoms < min_heavy_atoms, inorganics, mixtures.
3. PAINS → flagged but kept (configurable via ``curate.pains_flag_keep``).
4. Aggregate replicates by InChIKey → median pIC50, drop where σ > threshold.
5. Add a binary ``active`` label at ``active_pchembl_threshold``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

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
    from rdkit import Chem  # local import keeps stage 0 unit-test importable

    return Chem


def _import_descriptors() -> Any:
    from rdkit.Chem import Descriptors

    return Descriptors


def _pains_catalog() -> Any:
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

    p = FilterCatalogParams()
    p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(p)


def standardize_smiles(smiles: str) -> str | None:
    """Run the ChEMBL Structure Pipeline standardize → parent → SMILES."""
    Chem = _import_rdkit()
    try:
        from chembl_structure_pipeline import standardizer  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised on missing dep
        raise RuntimeError("chembl_structure_pipeline is required for curation") from exc

    if not isinstance(smiles, str) or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        std = standardizer.standardize_mol(mol)
        parent, _ = standardizer.get_parent_mol(std)
        if parent is None:
            return None
        return Chem.MolToSmiles(parent)
    except Exception:
        return None


def is_mixture(smiles: str) -> bool:
    return "." in smiles


def is_inorganic_mol(mol: Any) -> bool:
    return not any(atom.GetSymbol() == "C" for atom in mol.GetAtoms())


def passes_property_filters(mol: Any, params: Params) -> tuple[bool, str | None]:
    Descriptors = _import_descriptors()
    mw = float(Descriptors.MolWt(mol))
    if mw < params.curate.mw_min or mw > params.curate.mw_max:
        return False, f"mw_oor:{mw:.1f}"
    if mol.GetNumHeavyAtoms() < params.curate.min_heavy_atoms:
        return False, "heavy_atoms_low"
    if params.curate.reject_inorganic and is_inorganic_mol(mol):
        return False, "inorganic"
    return True, None


def aggregate_replicates(df: pd.DataFrame, sigma_threshold: float) -> pd.DataFrame:
    """Median-aggregate by InChIKey; drop groups whose σ exceeds the threshold."""
    Chem = _import_rdkit()

    def _inchikey(smi: str) -> str | None:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(mol) if mol is not None else None

    df = df.copy()
    df["inchikey"] = df["canonical_smiles_std"].map(_inchikey)
    df = df.dropna(subset=["inchikey"])

    rows: list[pd.Series] = []
    dropped_high_sigma = 0
    for _, g in df.groupby("inchikey", sort=False):
        if len(g) > 1 and float(g["pchembl_value"].std(ddof=0)) > sigma_threshold:
            dropped_high_sigma += 1
            continue
        rep = g.iloc[0].copy()
        rep["pchembl_value"] = float(g["pchembl_value"].median())
        rep["n_replicates"] = int(len(g))
        rows.append(rep)
    out = pd.DataFrame(rows).reset_index(drop=True)
    logger.info(
        "replicate aggregation: kept=%d dropped_high_sigma=%d", len(out), dropped_high_sigma
    )
    return out


def curate(raw: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Apply the full Stage 2 curation pipeline to a raw extract frame."""
    Chem = _import_rdkit()

    df = raw.copy()
    df["canonical_smiles_std"] = df["canonical_smiles"].map(standardize_smiles)
    df = df.dropna(subset=["canonical_smiles_std"]).reset_index(drop=True)

    keep_idx: list[int] = []
    reject_log: list[tuple[int, str]] = []
    for i, smi in enumerate(df["canonical_smiles_std"]):
        if params.curate.reject_mixtures and is_mixture(smi):
            reject_log.append((i, "mixture"))
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            reject_log.append((i, "parse_failed"))
            continue
        ok, reason = passes_property_filters(mol, params)
        if not ok:
            reject_log.append((i, reason or "rejected"))
            continue
        keep_idx.append(i)
    df = df.iloc[keep_idx].reset_index(drop=True)
    logger.info("property filters: kept=%d rejected=%d", len(df), len(reject_log))

    catalog = _pains_catalog()
    pains_flags: list[bool] = []
    for smi in df["canonical_smiles_std"]:
        mol = Chem.MolFromSmiles(smi)
        pains_flags.append(bool(catalog.HasMatch(mol)) if mol is not None else False)
    df["pains_flag"] = pains_flags
    if not params.curate.pains_flag_keep:
        df = df[~df["pains_flag"]].reset_index(drop=True)

    df = aggregate_replicates(df, params.curate.replicate_sigma_threshold)
    df["active"] = df["pchembl_value"] >= params.curate.active_pchembl_threshold
    return df


def run_curate(in_path: Path, out_dir: Path, params: Params) -> Path:
    """End-to-end Stage 2: parquet in → curated parquet + manifest out."""
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(in_path)
    curated = curate(raw, params)

    out_path = out_dir / "brd4_curated.parquet"
    curated.to_parquet(out_path, index=False)

    run_dir = make_run_dir(out_dir)
    manifest = Manifest(
        stage="curate",
        git_sha=get_git_sha(),
        timestamp=utc_timestamp(),
        inputs={"raw_parquet": file_signature(in_path)},
        outputs={
            "curated_parquet": file_signature(out_path),
            "n_rows_in": int(len(raw)),
            "n_rows_out": int(len(curated)),
            "n_active": int(curated["active"].sum()) if "active" in curated.columns else 0,
        },
        params_snapshot=params.model_dump(),
        seeds={"global": params.seed},
        env=env_snapshot(),
        wall_time_seconds=round(time.perf_counter() - started, 4),
    )
    write_manifest(manifest, run_dir)
    return out_path
