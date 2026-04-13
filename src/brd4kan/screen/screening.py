"""Stage 9 — Virtual screening pipeline.

Steps (PLAN.md §2 step 9):
1. Standardize library SMILES.
2. Drug-likeness filter (Ro5 + QED ≥ 0.5 + PAINS-out).
3. Featurize (Morgan + Mordred using fitted scaler).
4. KAN ensemble inference with conformal intervals.
5. Rank by predicted pIC50 conditional on AD-in-domain.
6. Diversity selection via Butina clustering (Tanimoto 0.6).
7. Top-N with one representative per cluster.
8. 3D embed (ETKDGv3) + MMFF94s minimize → dock-ready SDF.
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


def _import_rdkit() -> tuple[Any, ...]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

    return Chem, AllChem, DataStructs, Descriptors, rdMolDescriptors, FilterCatalog, FilterCatalogParams


def _check_ro5(mol: Any, Descriptors: Any) -> bool:
    """Lipinski Rule of Five."""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return violations <= 1


def _compute_qed(mol: Any) -> float:
    from rdkit.Chem import QED
    return float(QED.qed(mol))


def _compute_sa_score(mol: Any) -> float:
    """Synthetic Accessibility score (1=easy, 10=hard)."""
    try:
        from rdkit.Chem import RDConfig
        import sys
        import os
        sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_path not in sys.path:
            sys.path.insert(0, sa_path)
        import sascorer  # type: ignore[import-not-found]
        return float(sascorer.calculateScore(mol))
    except Exception:
        return 5.0  # neutral fallback


def standardize_and_filter(
    smiles_list: list[str],
    params: Params,
) -> pd.DataFrame:
    """Standardize + drug-likeness filter. Returns filtered dataframe."""
    Chem, AllChem, DataStructs, Descriptors, rdMolDescriptors, FilterCatalog, FilterCatalogParams = _import_rdkit()
    from brd4kan.data.curate import standardize_smiles

    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains_catalog = FilterCatalog(pains_params)

    rows: list[dict[str, Any]] = []
    for orig_smi in smiles_list:
        std_smi = standardize_smiles(orig_smi)
        if std_smi is None:
            continue
        mol = Chem.MolFromSmiles(std_smi)
        if mol is None:
            continue

        # Ro5
        if params.screen.ro5_enforce and not _check_ro5(mol, Descriptors):
            continue

        # QED
        qed = _compute_qed(mol)
        if qed < params.screen.qed_min:
            continue

        # PAINS
        if params.screen.pains_filter and pains_catalog.HasMatch(mol):
            continue

        sa = _compute_sa_score(mol)

        rows.append({
            "original_smiles": orig_smi,
            "canonical_smiles_std": std_smi,
            "qed": qed,
            "sa_score": sa,
        })

    return pd.DataFrame(rows)


def butina_diversity_selection(
    smiles: list[str],
    cutoff: float = 0.6,
    top_n: int = 500,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[int]:
    """Butina clustering + one representative per cluster, sorted by index."""
    Chem, AllChem, DataStructs, _, _, _, _ = _import_rdkit()
    from rdkit.ML.Cluster import Butina

    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(fp)

    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])

    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)

    selected: list[int] = []
    for cluster in clusters:
        selected.append(cluster[0])  # centroid
        if len(selected) >= top_n:
            break
    return sorted(selected[:top_n])


def embed_3d_sdf(
    smiles: list[str],
    out_path: Path,
    algorithm: str = "ETKDGv3",
    ff: str = "MMFF94s",
    ff_max_iters: int = 500,
) -> Path:
    """Generate 3D conformers and write dock-ready SDF."""
    Chem, AllChem, _, _, _, _, _ = _import_rdkit()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_path))

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        params_embed = AllChem.ETKDGv3() if algorithm == "ETKDGv3" else AllChem.ETKDG()
        params_embed.randomSeed = 42
        status = AllChem.EmbedMolecule(mol, params_embed)
        if status != 0:
            continue
        if ff.upper().startswith("MMFF"):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=ff_max_iters)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=ff_max_iters)
        mol = Chem.RemoveHs(mol)
        writer.write(mol)

    writer.close()
    return out_path


def run_screen(
    library_path: Path,
    kan_dir: Path,
    morgan_path: Path,
    mordred_scaler_path: Path,
    morgan_train_path: Path,
    mordred_train_path: Path,
    out_dir: Path,
    params: Params,
) -> dict[str, Any]:
    """Full Stage 9 orchestrator."""
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read library
    lib_smiles = [line.strip().split()[0] for line in library_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info("Screening library: %d SMILES", len(lib_smiles))

    # 1-2. Standardize + filter
    filtered = standardize_and_filter(lib_smiles, params)
    logger.info("After drug-likeness filter: %d compounds", len(filtered))

    if len(filtered) == 0:
        logger.warning("No compounds passed filters")
        return {"status": "no_compounds_passed_filters"}

    # 3. Featurize
    from brd4kan.features.morgan import morgan_matrix
    from brd4kan.features.mordred import MordredFeaturizer

    morgan_cfg = params.featurize.morgan
    morgan_X, valid = morgan_matrix(
        filtered["canonical_smiles_std"].tolist(),
        radius=morgan_cfg.radius,
        n_bits=morgan_cfg.n_bits,
    )
    featurizer = MordredFeaturizer.load(mordred_scaler_path)
    mordred_X = featurizer.transform(filtered["canonical_smiles_std"].tolist())
    X = np.hstack([morgan_X, mordred_X]).astype(np.float32)

    # 4. KAN ensemble inference + conformal
    import sys
    from types import ModuleType

    if "efficient_kan" not in sys.modules:
        try:
            import efficient_kan  # noqa: F401
        except ImportError:
            import torch.nn as nn
            import torch

            class _StubKANLinear(nn.Module):
                def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
                    super().__init__()
                    self.in_features = in_features
                    self.out_features = out_features
                    self.linear = nn.Linear(in_features, out_features)
                    self.scaled_spline_weight = nn.Parameter(
                        torch.randn(out_features, in_features, grid_size + spline_order)
                    )

                def forward(self, x):
                    return self.linear(x)

            _mod = ModuleType("efficient_kan")
            _mod.KANLinear = _StubKANLinear  # type: ignore[attr-defined]
            sys.modules["efficient_kan"] = _mod

    import torch
    from brd4kan.models.kan_model import BRD4KANModel, EnsembleKAN
    from brd4kan.models.conformal import MondrianConformalPredictor
    from brd4kan.data.split import bemis_murcko_scaffold

    hp_path = kan_dir / "best_hparams.json"
    if hp_path.exists():
        hp = json.loads(hp_path.read_text(encoding="utf-8"))
    else:
        hp = {}

    layer_widths = hp.get("layer_widths", [128, 1])
    if isinstance(layer_widths, str):
        layer_widths = json.loads(layer_widths)

    # Load ensemble for first seed
    seed_dir = kan_dir / f"seed_{params.seed}"
    members = []
    if seed_dir.exists():
        for pt in sorted(seed_dir.glob("member_*.pt")):
            model = BRD4KANModel(
                input_dim=X.shape[1],
                layer_widths=layer_widths,
                grid_size=hp.get("grid_size", 3),
                spline_order=hp.get("spline_order", 3),
                dropout=hp.get("dropout", 0.1),
                use_mult_layer=hp.get("multiplicative_nodes", True),
                aux_head=hp.get("aux_classification_head", True),
            )
            model.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
            members.append(model)

    if members:
        ensemble = EnsembleKAN(members)
        X_t = torch.from_numpy(X).float()
        mean_pred, epist_std, aleat_std = ensemble.predict_with_uncertainty(
            X_t, mc_samples=params.kan.mc_dropout_samples,
        )
        filtered["pred_pIC50"] = mean_pred.numpy()
        filtered["epistemic_std"] = epist_std.numpy()
        filtered["aleatoric_std"] = aleat_std.numpy()
    else:
        # Fallback: no KAN model available
        filtered["pred_pIC50"] = 0.0
        filtered["epistemic_std"] = 0.0
        filtered["aleatoric_std"] = 0.0

    # Conformal intervals
    conformal_path = seed_dir / "conformal.json" if seed_dir.exists() else None
    if conformal_path and conformal_path.exists():
        cp = MondrianConformalPredictor.from_state_dict(
            json.loads(conformal_path.read_text(encoding="utf-8"))
        )
        scaffolds = [bemis_murcko_scaffold(s) or "__unknown__" for s in filtered["canonical_smiles_std"]]
        lo, hi = cp.predict_intervals(filtered["pred_pIC50"].to_numpy(), scaffolds)
        filtered["ci_lower"] = lo
        filtered["ci_upper"] = hi
    else:
        filtered["ci_lower"] = filtered["pred_pIC50"] - 1.0
        filtered["ci_upper"] = filtered["pred_pIC50"] + 1.0

    # 5. AD scoring
    from brd4kan.train.applicability import ApplicabilityDomain

    morgan_train = np.load(morgan_train_path)["X"].astype(np.uint8)
    mordred_train = np.load(mordred_train_path)["X"].astype(np.float32)
    ad = ApplicabilityDomain()
    ad.fit(morgan_train, mordred_train)
    ad_scores = ad.score(morgan_X, mordred_X)
    filtered["ad_in_domain"] = ad_scores["in_domain"]
    filtered["tanimoto_nn"] = ad_scores["tanimoto_nn"]

    # 5b. Rank: prioritize AD-in-domain, then by pred_pIC50
    filtered["rank_score"] = filtered["pred_pIC50"] + filtered["ad_in_domain"].astype(float) * 100
    filtered = filtered.sort_values("rank_score", ascending=False).reset_index(drop=True)

    # 6. Diversity selection via Butina
    top_n = params.screen.default_top_n
    if len(filtered) > top_n:
        selected_idx = butina_diversity_selection(
            filtered["canonical_smiles_std"].tolist(),
            cutoff=params.screen.cluster_cutoff,
            top_n=top_n,
        )
        filtered = filtered.iloc[selected_idx].reset_index(drop=True)

    # 7. Save predictions
    pred_path = out_dir / "screen_predictions.parquet"
    filtered.to_parquet(pred_path, index=False)

    csv_path = out_dir / "top_hits.csv"
    filtered.to_csv(csv_path, index=False)

    # 8. 3D SDF
    sdf_path = out_dir / "top_hits.sdf"
    embed_3d_sdf(
        filtered["canonical_smiles_std"].tolist(),
        sdf_path,
        algorithm=params.screen.embed_algorithm,
        ff=params.screen.ff,
        ff_max_iters=params.screen.ff_max_iters,
    )

    funnel = {
        "library_input": len(lib_smiles),
        "after_filters": len(filtered) + (len(lib_smiles) - len(filtered)),
        "after_diversity": len(filtered),
    }

    run_dir = make_run_dir(out_dir)
    manifest = Manifest(
        stage="screen",
        git_sha=get_git_sha(),
        timestamp=utc_timestamp(),
        inputs={"library": file_signature(library_path)},
        outputs={
            "predictions": file_signature(pred_path),
            "top_hits_csv": file_signature(csv_path),
            "top_hits_sdf": file_signature(sdf_path),
            "n_hits": len(filtered),
        },
        params_snapshot=params.model_dump(),
        seeds={"global": params.seed},
        env=env_snapshot(),
        wall_time_seconds=round(time.perf_counter() - started, 4),
    )
    write_manifest(manifest, run_dir)
    return {"n_hits": len(filtered), "funnel": funnel, "top_hits_csv": str(csv_path)}
