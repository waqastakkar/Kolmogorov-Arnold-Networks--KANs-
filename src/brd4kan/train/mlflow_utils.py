"""MLflow logging helpers — thin wrappers so every stage uses a consistent setup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Set the tracking URI and create/set the experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_run(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: dict[str, Path] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log a single MLflow run and return the run_id."""
    with mlflow.start_run(run_name=run_name) as run:
        flat_params = _flatten(params)
        # MLflow has a 500 param limit, truncate keys > 250 chars
        for k, v in flat_params.items():
            mlflow.log_param(k[:250], v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        if tags:
            mlflow.set_tags(tags)
        if artifacts:
            for label, path in artifacts.items():
                if path.exists():
                    mlflow.log_artifact(str(path), artifact_path=label)
        return run.info.run_id


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out
