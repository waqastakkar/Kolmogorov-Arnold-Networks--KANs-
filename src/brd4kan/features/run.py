"""Stage 4 orchestrator — run Morgan + Mordred + graph featurization end-to-end."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from brd4kan.features.graphs import save_chemprop_csv
from brd4kan.features.mordred import MordredFeaturizer
from brd4kan.features.morgan import morgan_matrix
from brd4kan.utils.config import Params
from brd4kan.utils.hashing import array_sha256, file_signature
from brd4kan.utils.manifest import (
    Manifest,
    env_snapshot,
    get_git_sha,
    utc_timestamp,
    write_manifest,
)
from brd4kan.utils.runs import make_run_dir


def run_featurize(
    curated_path: Path,
    scaffold_split_path: Path,
    out_dir: Path,
    params: Params,
) -> dict[str, Path]:
    """Featurize the curated dataset into Morgan / Mordred / Chemprop caches."""
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(curated_path)
    smiles = df["canonical_smiles_std"].tolist()
    targets = df["pchembl_value"].astype(float).tolist()

    splits = json.loads(Path(scaffold_split_path).read_text(encoding="utf-8"))
    train_idx = splits["train"]

    morgan_cfg = params.featurize.morgan
    morgan, valid_morgan = morgan_matrix(
        smiles,
        radius=morgan_cfg.radius,
        n_bits=morgan_cfg.n_bits,
        use_features=morgan_cfg.use_features,
        use_chirality=morgan_cfg.use_chirality,
    )
    morgan_path = out_dir / "morgan.npz"
    np.savez_compressed(morgan_path, X=morgan, valid=valid_morgan)
    morgan_hash = array_sha256(morgan)

    mordred_cfg = params.featurize.mordred
    featurizer = MordredFeaturizer(
        ignore_3d=mordred_cfg.ignore_3d,
        variance_threshold=mordred_cfg.variance_threshold,
        correlation_threshold=mordred_cfg.correlation_threshold,
    )
    train_smiles = [smiles[i] for i in train_idx]
    featurizer.fit_transform(train_smiles)
    mordred_X = featurizer.transform(smiles)
    mordred_path = out_dir / "mordred.npz"
    np.savez_compressed(mordred_path, X=mordred_X, columns=np.array(featurizer.kept_columns))
    scaler_path = out_dir / "mordred_scaler.joblib"
    featurizer.save(scaler_path)
    mordred_hash = array_sha256(mordred_X)

    graph_path = out_dir / "chemprop.csv"
    save_chemprop_csv(smiles, targets, graph_path)

    run_dir = make_run_dir(out_dir)
    manifest = Manifest(
        stage="featurize",
        git_sha=get_git_sha(),
        timestamp=utc_timestamp(),
        inputs={
            "curated_parquet": file_signature(curated_path),
            "scaffold_split_json": file_signature(scaffold_split_path),
        },
        outputs={
            "morgan_npz": file_signature(morgan_path),
            "morgan_array_sha256": morgan_hash,
            "morgan_shape": list(morgan.shape),
            "mordred_npz": file_signature(mordred_path),
            "mordred_scaler": file_signature(scaler_path),
            "mordred_array_sha256": mordred_hash,
            "mordred_shape": list(mordred_X.shape),
            "mordred_kept_columns": len(featurizer.kept_columns or []),
            "chemprop_csv": file_signature(graph_path),
        },
        params_snapshot=params.model_dump(),
        seeds={"global": params.seed},
        env=env_snapshot(),
        wall_time_seconds=round(time.perf_counter() - started, 4),
    )
    write_manifest(manifest, run_dir)
    return {
        "morgan": morgan_path,
        "mordred": mordred_path,
        "mordred_scaler": scaler_path,
        "chemprop": graph_path,
    }
