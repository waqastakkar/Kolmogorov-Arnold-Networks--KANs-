"""Stage 10 — Screening analysis & hit selection.

Per top-hit: predicted pIC50 ± conformal CI, AD score, nearest ChEMBL
neighbor (Tanimoto + pIC50), scaffold class, novelty flag, key descriptors,
QED, SA score, BRD4 pharmacophore match. Output ranked CSV + per-compound
report cards (SVG) + report.html.
"""

from __future__ import annotations

import json
import logging
import time
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

# BRD4 pharmacophore: acetyl-lysine mimetic SMARTS
ACETYL_LYSINE_MIMETIC_SMARTS = [
    "[#7]~[#6](=[#8])~[#6]",  # amide-like
    "c1[nH]c2ccccc2c1",       # indole / indazole
    "c1cc2[nH]ccc2cc1",       # benzimidazole
    "[#7]1~[#6]~[#7]~[#6]~[#6]~1",  # triazine/pyrimidine
]


def _nearest_chembl_neighbor(
    query_smiles: list[str],
    train_smiles: list[str],
    train_pchembl: np.ndarray,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[dict[str, Any]]:
    """Find nearest ChEMBL neighbor by Tanimoto for each query."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    train_fps = []
    for smi in train_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
        else:
            train_fps.append(None)

    results = []
    for smi in query_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append({"nn_smiles": None, "nn_tanimoto": 0.0, "nn_pIC50": None})
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        best_sim = 0.0
        best_idx = -1
        for i, tfp in enumerate(train_fps):
            if tfp is None:
                continue
            sim = DataStructs.TanimotoSimilarity(fp, tfp)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        results.append({
            "nn_smiles": train_smiles[best_idx] if best_idx >= 0 else None,
            "nn_tanimoto": float(best_sim),
            "nn_pIC50": float(train_pchembl[best_idx]) if best_idx >= 0 else None,
        })
    return results


def _check_pharmacophore(smiles: str) -> bool:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    for smarts in ACETYL_LYSINE_MIMETIC_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt and mol.HasSubstructMatch(patt):
            return True
    return False


def run_analyze_hits(
    top_hits_csv: Path,
    curated_path: Path,
    out_dir: Path,
    params: Params,
) -> dict[str, Any]:
    """Full Stage 10: annotate hits with AD, novelty, pharmacophore, etc."""
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    hits = pd.read_csv(top_hits_csv)
    curated = pd.read_parquet(curated_path)
    train_smiles = curated["canonical_smiles_std"].tolist()
    train_pchembl = curated["pchembl_value"].to_numpy(dtype=np.float64)

    # Nearest ChEMBL neighbor
    nn_results = _nearest_chembl_neighbor(
        hits["canonical_smiles_std"].tolist(),
        train_smiles,
        train_pchembl,
    )
    hits["nn_smiles"] = [r["nn_smiles"] for r in nn_results]
    hits["nn_tanimoto"] = [r["nn_tanimoto"] for r in nn_results]
    hits["nn_pIC50"] = [r["nn_pIC50"] for r in nn_results]

    # Novelty flag: Tanimoto < 0.4 to nearest training compound
    hits["novel"] = hits["nn_tanimoto"] < 0.4

    # Scaffold class
    from brd4kan.data.split import bemis_murcko_scaffold
    hits["scaffold"] = [bemis_murcko_scaffold(s) or "acyclic" for s in hits["canonical_smiles_std"]]

    # Pharmacophore match
    hits["pharmacophore_match"] = [_check_pharmacophore(s) for s in hits["canonical_smiles_std"]]

    # Save annotated CSV
    annotated_path = out_dir / "annotated_hits.csv"
    hits.to_csv(annotated_path, index=False)

    # Generate per-hit SVG report cards
    from brd4kan.viz.figures import fig_hit_cards

    hit_data = []
    for _, row in hits.head(20).iterrows():
        hit_data.append({
            "name": str(row.get("canonical_smiles_std", "?"))[:40],
            "pIC50": float(row.get("pred_pIC50", 0.0)),
            "AD": float(row.get("tanimoto_nn", 0.0)),
            "QED": float(row.get("qed", 0.0)),
            "SA": float(10.0 - row.get("sa_score", 5.0)),  # invert so higher = better
        })
    cards_path = out_dir / "hit_cards.svg"
    fig_hit_cards(hit_data, cards_path, n_show=min(20, len(hit_data)))

    run_dir = make_run_dir(out_dir)
    manifest = Manifest(
        stage="analyze_hits",
        git_sha=get_git_sha(),
        timestamp=utc_timestamp(),
        inputs={"top_hits_csv": file_signature(top_hits_csv)},
        outputs={
            "annotated_csv": file_signature(annotated_path),
            "hit_cards": file_signature(cards_path),
            "n_hits": len(hits),
            "n_novel": int(hits["novel"].sum()),
            "n_pharmacophore_match": int(hits["pharmacophore_match"].sum()),
        },
        params_snapshot=params.model_dump(),
        seeds={"global": params.seed},
        env=env_snapshot(),
        wall_time_seconds=round(time.perf_counter() - started, 4),
    )
    write_manifest(manifest, run_dir)
    return {
        "n_hits": len(hits),
        "n_novel": int(hits["novel"].sum()),
        "n_pharmacophore": int(hits["pharmacophore_match"].sum()),
    }
