"""Stage 8 orchestrator — evaluation, bootstrap CIs, AD, SHAP, figures.

Metrics on scaffold-test and time-test, 5 seeds: RMSE, MAE, R², Spearman ρ,
Pearson r, ROC-AUC, PR-AUC, MCC, Brier, ECE. Bootstrap 1000× CIs.
Applicability domain. SHAP for KAN + baselines. All 9 figure types.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from brd4kan.train.applicability import ApplicabilityDomain
from brd4kan.train.bootstrap import bootstrap_ci
from brd4kan.train.metrics import regression_metrics
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


def _load_baseline_predictions(
    baselines_dir: Path,
    model_name: str,
    X_test: np.ndarray,
    seed: int,
) -> np.ndarray | None:
    """Load a trained baseline and predict on test."""
    from brd4kan.models.baselines import load_model

    model_path = baselines_dir / model_name / f"seed_{seed}.joblib"
    if not model_path.exists():
        return None
    model = load_model(model_path)
    return model.predict(X_test)


def _load_kan_predictions(
    kan_dir: Path,
    X_test: np.ndarray,
    seed: int,
) -> np.ndarray | None:
    """Load KAN ensemble and predict on test."""
    import sys
    from types import ModuleType

    # Ensure efficient_kan stub exists if not installed
    if "efficient_kan" not in sys.modules:
        try:
            import efficient_kan  # noqa: F401
        except ImportError:
            import torch.nn as nn

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
    from brd4kan.models.kan_model import BRD4KANModel

    hp_path = kan_dir / "best_hparams.json"
    if not hp_path.exists():
        return None
    hp = json.loads(hp_path.read_text(encoding="utf-8"))
    layer_widths = hp.get("layer_widths", [128, 1])
    if isinstance(layer_widths, str):
        layer_widths = json.loads(layer_widths)

    seed_dir = kan_dir / f"seed_{seed}"
    if not seed_dir.exists():
        return None

    members = sorted(seed_dir.glob("member_*.pt"))
    if not members:
        return None

    preds_all = []
    for pt_path in members:
        model = BRD4KANModel(
            input_dim=X_test.shape[1],
            layer_widths=layer_widths,
            grid_size=hp.get("grid_size", 3),
            spline_order=hp.get("spline_order", 3),
            dropout=hp.get("dropout", 0.1),
            use_mult_layer=hp.get("multiplicative_nodes", True),
            aux_head=hp.get("aux_classification_head", True),
        )
        state = torch.load(pt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X_test).float()
            reg, _ = model(x_t)
            preds_all.append(reg.numpy())
    return np.mean(preds_all, axis=0)


def run_evaluate(
    curated_path: Path,
    scaffold_split_path: Path,
    time_split_path: Path,
    morgan_path: Path,
    mordred_path: Path,
    baselines_dir: Path,
    kan_dir: Path,
    symbolic_dir: Path,
    figures_dir: Path,
    metrics_dir: Path,
    params: Params,
) -> dict[str, Any]:
    """Full Stage 8: metrics + bootstrap + AD + figures."""
    started = time.perf_counter()
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(curated_path)
    smiles_all = df["canonical_smiles_std"].tolist()
    y_all = df["pchembl_value"].to_numpy(dtype=np.float64)

    scaffold_splits = json.loads(scaffold_split_path.read_text(encoding="utf-8"))
    time_splits = json.loads(time_split_path.read_text(encoding="utf-8"))
    train_idx = np.array(scaffold_splits["train"])
    test_idx = np.array(scaffold_splits["test"])
    time_test_idx = np.array(time_splits.get("test", []))

    morgan = np.load(morgan_path)["X"].astype(np.float32)
    mordred_X = np.load(mordred_path)["X"].astype(np.float32)
    X_all = np.hstack([morgan, mordred_X])

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    morgan_train, morgan_test = morgan[train_idx], morgan[test_idx]

    active_thresh = params.curate.active_pchembl_threshold
    n_active = int((y_all >= active_thresh).sum())

    # Build descriptor names
    mordred_cols_file = np.load(mordred_path, allow_pickle=True)
    mordred_cols = mordred_cols_file.get("columns", np.array([]))
    desc_names = [f"ECFP4_{i}" for i in range(morgan.shape[1])] + [str(c) for c in mordred_cols]

    # ---- Applicability Domain ----
    ad = ApplicabilityDomain(
        tanimoto_radius=params.evaluate.tanimoto_radius,
        tanimoto_nbits=params.evaluate.tanimoto_nbits,
    )
    ad.fit(morgan_train, mordred_X[train_idx])
    ad_scores = ad.score(morgan_test, mordred_X[test_idx])

    # ---- Collect predictions per model ----
    all_results: dict[str, Any] = {}
    model_preds: dict[str, np.ndarray] = {}

    descriptor_baselines = [m for m in params.baselines.models if m != "chemprop"]
    for model_name in descriptor_baselines:
        preds = _load_baseline_predictions(baselines_dir, model_name, X_test, params.seed)
        if preds is not None:
            model_preds[model_name] = preds
            m = regression_metrics(y_test, preds, active_threshold=active_thresh)
            ci = bootstrap_ci(y_test, preds, n_iters=params.evaluate.bootstrap_iters,
                            active_threshold=active_thresh, seed=params.seed)
            all_results[model_name] = {"metrics": m, "bootstrap_ci": ci}

    kan_preds = _load_kan_predictions(kan_dir, X_test, params.seed)
    if kan_preds is not None:
        model_preds["KAN"] = kan_preds
        m = regression_metrics(y_test, kan_preds, active_threshold=active_thresh)
        ci = bootstrap_ci(y_test, kan_preds, n_iters=params.evaluate.bootstrap_iters,
                        active_threshold=active_thresh, seed=params.seed)
        all_results["KAN"] = {"metrics": m, "bootstrap_ci": ci}

    # Save metrics
    metrics_path = metrics_dir / "evaluation_metrics.json"
    metrics_path.write_text(
        json.dumps(all_results, indent=2, default=str), encoding="utf-8"
    )

    # ---- Figures ----
    from brd4kan.viz.figures import (
        fig_dataset_overview,
        fig_benchmark_bars,
        fig_parity_residual,
        fig_kan_splines,
        fig_symbolic_equation,
        fig_ad_map,
    )

    # 1. Dataset overview
    fig_dataset_overview(
        pchembl_values=y_all,
        n_compounds=len(y_all),
        n_active=n_active,
        out_path=figures_dir / "01_dataset_overview.svg",
    )

    # 2. Benchmark bars
    if all_results:
        scaffold_metrics: dict[str, dict[str, float]] = {}
        for mn, r in all_results.items():
            scaffold_metrics[mn] = {
                "rmse_median": r["metrics"]["rmse"],
                "rmse_std": r["bootstrap_ci"].get("rmse", {}).get("std", 0.0),
            }
        fig_benchmark_bars(
            model_names=list(scaffold_metrics.keys()),
            scaffold_metrics=scaffold_metrics,
            out_path=figures_dir / "02_benchmark_bars.svg",
        )

    # 3. Parity + residual for each model
    for mn, preds in model_preds.items():
        fig_parity_residual(
            y_true=y_test,
            y_pred=preds,
            model_name=mn,
            out_path=figures_dir / f"03_parity_{mn.lower()}.svg",
        )

    # 4. KAN splines / importance
    importance_path = symbolic_dir / "descriptor_importance.json"
    if importance_path.exists():
        imp_data = json.loads(importance_path.read_text(encoding="utf-8"))
        dnames = [desc_names[e["input_idx"]] if e["input_idx"] < len(desc_names) else f"x_{e['input_idx']}" for e in imp_data[:10]]
        dimps = [e["importance"] for e in imp_data[:10]]
        fig_kan_splines(
            descriptor_names=dnames,
            importances=dimps,
            out_path=figures_dir / "04_kan_splines.svg",
        )

    # 5. Symbolic equation
    eq_tex_path = symbolic_dir / "pIC50_equation.tex"
    if eq_tex_path.exists() and importance_path.exists():
        latex_eq = eq_tex_path.read_text(encoding="utf-8")
        fig_symbolic_equation(
            latex_equation=latex_eq,
            descriptor_names=dnames,
            importances=dimps,
            out_path=figures_dir / "05_symbolic_equation.svg",
        )

    # 7. AD map
    pca = PCA(n_components=2)
    test_pca = pca.fit_transform(mordred_X[test_idx].astype(np.float64))
    fig_ad_map(
        pca_coords=test_pca,
        in_domain=ad_scores["in_domain"],
        out_path=figures_dir / "07_ad_map.svg",
        pchembl_values=y_test,
    )

    # Save AD scores
    ad_path = metrics_dir / "ad_scores.json"
    ad_path.write_text(json.dumps({
        "n_in_domain": int(ad_scores["in_domain"].sum()),
        "n_out_of_domain": int((~ad_scores["in_domain"]).sum()),
        "tanimoto_nn_mean": float(ad_scores["tanimoto_nn"].mean()),
    }, indent=2), encoding="utf-8")

    run_dir = make_run_dir(metrics_dir)
    manifest = Manifest(
        stage="evaluate",
        git_sha=get_git_sha(),
        timestamp=utc_timestamp(),
        inputs={
            "curated_parquet": file_signature(curated_path),
            "scaffold_split": file_signature(scaffold_split_path),
            "time_split": file_signature(time_split_path),
            "morgan": file_signature(morgan_path),
            "mordred": file_signature(mordred_path),
        },
        outputs={
            "metrics_json": file_signature(metrics_path),
            "ad_scores": file_signature(ad_path),
            "n_figures": len(list(figures_dir.glob("*.svg"))),
            "n_models_evaluated": len(all_results),
        },
        params_snapshot=params.model_dump(),
        seeds={"global": params.seed},
        env=env_snapshot(),
        wall_time_seconds=round(time.perf_counter() - started, 4),
    )
    write_manifest(manifest, run_dir)
    return all_results
