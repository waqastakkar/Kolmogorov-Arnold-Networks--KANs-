"""Stage 5 orchestrator — Optuna-tuned baselines with MLflow logging.

For each baseline model (RF, XGBoost, SVR, MLP):
1. Optuna 100 trials, 5-fold scaffold CV inside train, minimize RMSE.
2. Best config retrained on full train with 5 seeds.
3. Evaluate each seed on scaffold-test → median + per-seed metrics.
4. Save model weights, Optuna study, MLflow run.

Chemprop is handled separately because it trains from SMILES, not
pre-computed feature matrices.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.pruners import HyperbandPruner

from brd4kan.models.baselines import (
    SUGGEST_FNS,
    create_model,
    save_model,
)
from brd4kan.train.cv import scaffold_cv_indices
from brd4kan.train.metrics import regression_metrics
from brd4kan.train.mlflow_utils import log_run, setup_mlflow
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


def _make_objective(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_smiles: list[str],
    cv_folds: int,
    seed: int,
) -> Any:
    """Return an Optuna objective that does scaffold CV and returns mean RMSE."""
    folds = scaffold_cv_indices(train_smiles, cv_folds)

    def objective(trial: Any) -> float:
        hparams = SUGGEST_FNS[model_name](trial)
        rmses: list[float] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            model = create_model(model_name, hparams, seed)
            model.fit(X_train[tr_idx], y_train[tr_idx])
            preds = model.predict(X_train[va_idx])
            rmse = float(np.sqrt(np.mean((y_train[va_idx] - preds) ** 2)))
            rmses.append(rmse)
            # Report intermediate for Hyperband pruning
            trial.report(rmse, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(rmses))

    return objective


def _train_and_evaluate_seeds(
    model_name: str,
    hparams: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_seeds: int,
    base_seed: int,
    model_dir: Path,
    active_threshold: float,
) -> dict[str, Any]:
    """Retrain best config with ``n_seeds`` seeds and evaluate on test."""
    all_metrics: list[dict[str, float]] = []
    for i in range(n_seeds):
        seed_i = base_seed + i
        set_global_seed(seed_i)
        model = create_model(model_name, hparams, seed_i)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        m = regression_metrics(y_test, preds, active_threshold=active_threshold)
        all_metrics.append(m)
        save_model(model, model_dir / f"seed_{seed_i}.joblib")

    # Aggregate: median across seeds
    agg: dict[str, float] = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        agg[f"{key}_median"] = float(np.nanmedian(vals))
        agg[f"{key}_std"] = float(np.nanstd(vals))
    return {"per_seed": all_metrics, "aggregated": agg}


def tune_baseline(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_smiles: list[str],
    params: Params,
    out_dir: Path,
) -> dict[str, Any]:
    """Full Optuna tune → multi-seed retrain → evaluate → log for one model."""
    n_trials = params.baselines.optuna_trials
    cv_folds = params.baselines.cv_folds
    n_seeds = params.baselines.n_seeds
    active_thresh = params.curate.active_pchembl_threshold

    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Optuna study
    study_path = out_dir / f"{model_name}_optuna.db"
    storage = f"sqlite:///{study_path}"
    study = optuna.create_study(
        study_name=model_name,
        storage=storage,
        direction="minimize",
        pruner=HyperbandPruner(),
        load_if_exists=True,
    )
    objective = _make_objective(
        model_name, X_train, y_train, train_smiles, cv_folds, params.seed
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_hparams = study.best_params
    best_cv_rmse = study.best_value
    logger.info(
        "%s Optuna done: best_cv_rmse=%.4f best_params=%s",
        model_name, best_cv_rmse, best_hparams,
    )

    # Save best hparams
    (model_dir / "best_hparams.json").write_text(
        json.dumps(best_hparams, indent=2, default=str), encoding="utf-8"
    )

    # Multi-seed retrain + test eval
    eval_result = _train_and_evaluate_seeds(
        model_name, best_hparams, X_train, y_train,
        X_test, y_test, n_seeds, params.seed, model_dir, active_thresh,
    )

    # Save test metrics
    metrics_path = model_dir / "test_metrics.json"
    metrics_path.write_text(
        json.dumps(eval_result, indent=2, default=str), encoding="utf-8"
    )

    # MLflow
    setup_mlflow(f"file:{params.paths.mlflow}", "baselines")
    mlflow_run_id = log_run(
        run_name=f"{model_name}_best",
        params={"model": model_name, **best_hparams},
        metrics=eval_result["aggregated"],
        artifacts={"model": model_dir},
        tags={"stage": "baselines", "model": model_name},
    )

    return {
        "model": model_name,
        "best_hparams": best_hparams,
        "best_cv_rmse": best_cv_rmse,
        "test_metrics": eval_result["aggregated"],
        "mlflow_run_id": mlflow_run_id,
    }


def _tune_chemprop(
    train_smiles: list[str],
    y_train: np.ndarray,
    test_smiles: list[str],
    y_test: np.ndarray,
    params: Params,
    out_dir: Path,
) -> dict[str, Any]:
    """Optuna-tuned Chemprop D-MPNN. Separate path because it uses raw SMILES."""
    from brd4kan.models.chemprop_wrapper import ChempropModel, suggest_chemprop

    n_trials = params.baselines.optuna_trials
    n_seeds = params.baselines.n_seeds
    active_thresh = params.curate.active_pchembl_threshold

    model_dir = out_dir / "chemprop"
    model_dir.mkdir(parents=True, exist_ok=True)

    cv_folds = scaffold_cv_indices(train_smiles, params.baselines.cv_folds)

    def objective(trial: Any) -> float:
        hparams = suggest_chemprop(trial)
        rmses: list[float] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(cv_folds):
            tr_smi = [train_smiles[i] for i in tr_idx]
            va_smi = [train_smiles[i] for i in va_idx]
            tr_y = y_train[tr_idx]
            va_y = y_train[va_idx]
            model = ChempropModel(hparams, params.seed)
            model.fit(tr_smi, tr_y)
            preds = model.predict(va_smi)
            rmse = float(np.sqrt(np.mean((va_y - preds) ** 2)))
            rmses.append(rmse)
            trial.report(rmse, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(rmses))

    study_path = out_dir / "chemprop_optuna.db"
    study = optuna.create_study(
        study_name="chemprop",
        storage=f"sqlite:///{study_path}",
        direction="minimize",
        pruner=HyperbandPruner(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_hparams = study.best_params
    best_cv_rmse = study.best_value

    (model_dir / "best_hparams.json").write_text(
        json.dumps(best_hparams, indent=2, default=str), encoding="utf-8"
    )

    all_metrics: list[dict[str, float]] = []
    for i in range(n_seeds):
        seed_i = params.seed + i
        set_global_seed(seed_i)
        model = ChempropModel(best_hparams, seed_i)
        model.fit(train_smiles, y_train)
        preds = model.predict(test_smiles)
        m = regression_metrics(y_test, preds, active_threshold=active_thresh)
        all_metrics.append(m)
        model.save(model_dir / f"seed_{seed_i}.pt")

    agg: dict[str, float] = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        agg[f"{key}_median"] = float(np.nanmedian(vals))
        agg[f"{key}_std"] = float(np.nanstd(vals))

    (model_dir / "test_metrics.json").write_text(
        json.dumps({"per_seed": all_metrics, "aggregated": agg}, indent=2, default=str),
        encoding="utf-8",
    )

    setup_mlflow(f"file:{params.paths.mlflow}", "baselines")
    mlflow_run_id = log_run(
        run_name="chemprop_best",
        params={"model": "chemprop", **best_hparams},
        metrics=agg,
        tags={"stage": "baselines", "model": "chemprop"},
    )

    return {
        "model": "chemprop",
        "best_hparams": best_hparams,
        "best_cv_rmse": best_cv_rmse,
        "test_metrics": agg,
        "mlflow_run_id": mlflow_run_id,
    }


def run_baselines(
    curated_path: Path,
    scaffold_split_path: Path,
    morgan_path: Path,
    mordred_path: Path,
    out_dir: Path,
    params: Params,
) -> dict[str, Any]:
    """Full Stage 5: tune + train + eval all baselines, with manifest."""
    import pandas as pd

    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(curated_path)
    smiles_all = df["canonical_smiles_std"].tolist()
    y_all = df["pchembl_value"].to_numpy(dtype=np.float64)

    splits = json.loads(Path(scaffold_split_path).read_text(encoding="utf-8"))
    train_idx = np.array(splits["train"])
    test_idx = np.array(splits["test"])

    morgan = np.load(morgan_path)["X"]
    mordred_X = np.load(mordred_path)["X"]
    X_desc = np.hstack([morgan, mordred_X])

    X_train, X_test = X_desc[train_idx], X_desc[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    train_smiles = [smiles_all[i] for i in train_idx]
    test_smiles = [smiles_all[i] for i in test_idx]

    results: dict[str, Any] = {}
    descriptor_models = [m for m in params.baselines.models if m != "chemprop"]

    for model_name in descriptor_models:
        logger.info("Tuning baseline: %s", model_name)
        results[model_name] = tune_baseline(
            model_name, X_train, y_train, X_test, y_test,
            train_smiles, params, out_dir,
        )

    if "chemprop" in params.baselines.models:
        logger.info("Tuning baseline: chemprop")
        try:
            results["chemprop"] = _tune_chemprop(
                train_smiles, y_train, test_smiles, y_test, params, out_dir,
            )
        except ImportError:
            logger.warning("Chemprop not installed — skipping D-MPNN baseline.")
            results["chemprop"] = {"status": "skipped", "reason": "chemprop not installed"}

    summary_path = out_dir / "baselines_summary.json"
    summary_path.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    run_dir = make_run_dir(out_dir)
    manifest = Manifest(
        stage="baselines",
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
            "n_models": len(results),
        },
        params_snapshot=params.model_dump(),
        seeds={"global": params.seed},
        env=env_snapshot(),
        wall_time_seconds=round(time.perf_counter() - started, 4),
    )
    write_manifest(manifest, run_dir)
    return results
