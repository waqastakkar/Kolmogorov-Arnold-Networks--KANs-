"""Stage 6 orchestrator — Optuna-tuned advanced KAN with ensemble + conformal.

1. Optuna 200 trials, multi-objective (RMSE ↓, sparsity ↑), Hyperband pruner.
2. Best Pareto-front config retrained with 5 seeds × ensemble_size members.
3. Mondrian conformal calibration on val split.
4. Evaluate ensemble on scaffold-test.
5. Save models, Optuna study, MLflow run.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from optuna.pruners import HyperbandPruner

from brd4kan.models.conformal import MondrianConformalPredictor
from brd4kan.models.kan_model import BRD4KANModel, EnsembleKAN
from brd4kan.train.cv import scaffold_cv_indices
from brd4kan.train.metrics import regression_metrics
from brd4kan.train.mlflow_utils import log_run, setup_mlflow
from brd4kan.train.train_kan import train_single_kan
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
from brd4kan.utils.seed import set_global_seed

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _get_device(params: Params) -> torch.device:
    import os

    dev_str = os.environ.get("BRD4KAN_DEVICE", "auto")
    if dev_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev_str)


def _suggest_kan_hparams(trial: Any, params: Params) -> dict[str, Any]:
    return {
        "layer_widths": trial.suggest_categorical(
            "layer_widths",
            ["[64,1]", "[128,1]", "[128,64,1]", "[256,64,1]"],
        ),
        "grid_size": trial.suggest_categorical("grid_size", [3, 5]),
        "spline_order": params.kan.spline_order,
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "lamb": trial.suggest_float("lamb", 1e-5, 1e-1, log=True),
        "lamb_entropy": trial.suggest_float("lamb_entropy", 0.1, 5.0, log=True),
        "lamb_coef": 0.0,
        "optimizer": trial.suggest_categorical("optimizer", ["adamw", "lbfgs"]),
        "epochs": 60,
        "grid_schedule": params.kan.grid_schedule,
        "early_stopping_patience": params.kan.early_stopping_patience,
        "grad_clip": params.kan.grad_clip,
        "multiplicative_nodes": params.kan.multiplicative_nodes,
        "aux_classification_head": params.kan.aux_classification_head,
    }


def _parse_layer_widths(s: str) -> list[int]:
    return json.loads(s)


def run_kan(
    curated_path: Path,
    scaffold_split_path: Path,
    morgan_path: Path,
    mordred_path: Path,
    out_dir: Path,
    params: Params,
) -> dict[str, Any]:
    """Full Stage 6: Optuna → multi-seed ensemble → conformal → eval."""
    import pandas as pd

    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _get_device(params)
    logger.info("KAN training on device: %s", device)

    # Load data
    df = pd.read_parquet(curated_path)
    smiles_all = df["canonical_smiles_std"].tolist()
    y_all = df["pchembl_value"].to_numpy(dtype=np.float64)

    splits = json.loads(Path(scaffold_split_path).read_text(encoding="utf-8"))
    train_idx = np.array(splits["train"])
    val_idx = np.array(splits["val"])
    test_idx = np.array(splits["test"])

    morgan = np.load(morgan_path)["X"].astype(np.float32)
    mordred_X = np.load(mordred_path)["X"].astype(np.float32)
    X_all = np.hstack([morgan, mordred_X])

    X_train, X_val, X_test = X_all[train_idx], X_all[val_idx], X_all[test_idx]
    y_train, y_val, y_test = y_all[train_idx], y_all[val_idx], y_all[test_idx]
    train_smiles = [smiles_all[i] for i in train_idx]
    val_smiles = [smiles_all[i] for i in val_idx]
    test_smiles = [smiles_all[i] for i in test_idx]

    active_thresh = params.curate.active_pchembl_threshold

    # ----- Optuna multi-objective study -----
    cv_folds = scaffold_cv_indices(train_smiles, params.kan.cv_folds)

    def objective(trial: Any) -> tuple[float, float]:
        hp = _suggest_kan_hparams(trial, params)
        hp["layer_widths"] = _parse_layer_widths(hp["layer_widths"])
        rmses: list[float] = []
        sparsities: list[float] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(cv_folds):
            model, hist = train_single_kan(
                X_train[tr_idx], y_train[tr_idx],
                X_train[va_idx], y_train[va_idx],
                hp, params.seed, device, active_thresh,
            )
            rmses.append(hist["best_val_rmse"][0])
            sparsities.append(hist["sparsity"][-1])
            trial.report(rmses[-1], fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(rmses)), -float(np.mean(sparsities))

    study_path = out_dir / "kan_optuna.db"
    study = optuna.create_study(
        study_name="kan",
        storage=f"sqlite:///{study_path}",
        directions=["minimize", "minimize"],  # RMSE ↓, -sparsity ↓ (=sparsity ↑)
        pruner=HyperbandPruner(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=params.kan.optuna_trials, show_progress_bar=False)

    # Pick best trial by RMSE among Pareto front
    pareto = study.best_trials
    best_trial = min(pareto, key=lambda t: t.values[0])
    best_hp = best_trial.params
    best_hp["layer_widths"] = _parse_layer_widths(best_hp["layer_widths"])
    # Fill in non-suggested params
    best_hp.update({
        "spline_order": params.kan.spline_order,
        "lamb_coef": 0.0,
        "epochs": 100,
        "grid_schedule": params.kan.grid_schedule,
        "early_stopping_patience": params.kan.early_stopping_patience,
        "grad_clip": params.kan.grad_clip,
        "multiplicative_nodes": params.kan.multiplicative_nodes,
        "aux_classification_head": params.kan.aux_classification_head,
    })

    (out_dir / "best_hparams.json").write_text(
        json.dumps(best_hp, indent=2, default=str), encoding="utf-8"
    )

    # ----- Multi-seed ensemble training -----
    from brd4kan.data.split import bemis_murcko_scaffold

    all_results: list[dict[str, Any]] = []
    for seed_offset in range(params.kan.n_seeds):
        seed_i = params.seed + seed_offset
        ensemble_members: list[BRD4KANModel] = []

        for ens_idx in range(params.kan.ensemble_size):
            member_seed = seed_i * 100 + ens_idx
            model, hist = train_single_kan(
                X_train, y_train, X_val, y_val,
                best_hp, member_seed, device, active_thresh,
            )
            ensemble_members.append(model)
            # Save individual member
            member_path = out_dir / f"seed_{seed_i}" / f"member_{ens_idx}.pt"
            member_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), member_path)

        ensemble = EnsembleKAN(ensemble_members)

        # ----- Conformal calibration on val -----
        val_scaffolds = [
            bemis_murcko_scaffold(s) or "__unknown__" for s in val_smiles
        ]
        X_val_t = torch.from_numpy(X_val).float().to(device)
        with torch.no_grad():
            val_pred = ensemble(X_val_t).cpu().numpy()
        val_residuals = y_val - val_pred

        conformal = MondrianConformalPredictor(alpha=params.conformal.alpha)
        conformal.calibrate(val_residuals, val_scaffolds)

        # ----- Test evaluation -----
        test_scaffolds = [
            bemis_murcko_scaffold(s) or "__unknown__" for s in test_smiles
        ]
        X_test_t = torch.from_numpy(X_test).float().to(device)
        mean_pred, epist_std, aleat_std = ensemble.predict_with_uncertainty(
            X_test_t, mc_samples=params.kan.mc_dropout_samples,
        )
        test_pred_np = mean_pred.cpu().numpy()
        m = regression_metrics(y_test, test_pred_np, active_threshold=active_thresh)
        cov = conformal.coverage(y_test, test_pred_np, test_scaffolds)
        m["conformal_coverage"] = cov["overall"]
        m["sparsity"] = float(ensemble_members[0].sparsity())

        # Save conformal
        conformal_path = out_dir / f"seed_{seed_i}" / "conformal.json"
        conformal_path.write_text(
            json.dumps(conformal.state_dict(), indent=2, default=str), encoding="utf-8"
        )

        all_results.append({
            "seed": seed_i,
            "metrics": m,
            "coverage": cov,
        })

    # Aggregate across seeds
    agg: dict[str, float] = {}
    for key in all_results[0]["metrics"]:
        vals = [r["metrics"][key] for r in all_results]
        agg[f"{key}_median"] = float(np.nanmedian(vals))
        agg[f"{key}_std"] = float(np.nanstd(vals))

    summary = {
        "best_hparams": best_hp,
        "best_optuna_rmse": best_trial.values[0],
        "best_optuna_sparsity": -best_trial.values[1],
        "per_seed": all_results,
        "aggregated": agg,
    }
    summary_path = out_dir / "kan_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # MLflow
    setup_mlflow(f"file:{params.paths.mlflow}", "kan")
    log_run(
        run_name="kan_best",
        params={"model": "kan", **{k: str(v) for k, v in best_hp.items()}},
        metrics=agg,
        tags={"stage": "kan"},
    )

    run_dir = make_run_dir(out_dir)
    manifest = Manifest(
        stage="kan",
        git_sha=get_git_sha(),
        timestamp=utc_timestamp(),
        inputs={
            "curated_parquet": file_signature(curated_path),
            "scaffold_split_json": file_signature(scaffold_split_path),
            "morgan_npz": file_signature(morgan_path),
            "mordred_npz": file_signature(mordred_path),
        },
        outputs={
            "summary": file_signature(summary_path),
            "study_db": file_signature(study_path),
        },
        params_snapshot=params.model_dump(),
        seeds={"global": params.seed},
        env=env_snapshot(),
        wall_time_seconds=round(time.perf_counter() - started, 4),
    )
    write_manifest(manifest, run_dir)
    return summary
