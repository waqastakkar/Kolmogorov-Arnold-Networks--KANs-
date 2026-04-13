"""End-to-end smoke test for Stage 1-4 orchestrators on tiny synthetic data.

Exercises ``run_extract`` (against ``tiny_chembl_db``), ``run_curate``,
``run_split`` and ``run_featurize`` so the canonical artifacts and per-run
manifests are generated under a temp directory. Asserts:

* Every canonical artifact path exists.
* Every run dir contains ``manifest.json`` + ``git_sha.txt``.
* The featurize step produces an array whose sha256 matches its manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

rdkit = pytest.importorskip("rdkit")
csp = pytest.importorskip("chembl_structure_pipeline")
mordred = pytest.importorskip("mordred")

from brd4kan.data.curate import run_curate  # noqa: E402
from brd4kan.data.extract import run_extract  # noqa: E402
from brd4kan.data.split import run_split  # noqa: E402
from brd4kan.features.run import run_featurize  # noqa: E402
from brd4kan.utils.config import Params  # noqa: E402


def _synthetic_curated_parquet(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        # Drug-like, varied scaffolds — large enough to satisfy the curate MW filter
        ("CHEMBL1", "Cc1cc(C)c2c(c1)C(=O)N(Cc1ccc(Cl)cc1)C2=O", 7.5, 2018),
        ("CHEMBL2", "Cc1cc(C)c2c(c1)C(=O)N(Cc1ccc(Br)cc1)C2=O", 7.0, 2019),
        ("CHEMBL3", "O=C1N(Cc2ccc(F)cc2)C(=O)c2cc(C)cc(C)c21", 6.5, 2020),
        ("CHEMBL4", "CN1CCN(c2ccc(NC(=O)c3ccc(Cl)cc3)cc2)CC1", 6.8, 2021),
        ("CHEMBL5", "Cn1ncc(-c2ccnc(N3CCCC3)n2)c1N", 7.2, 2017),
        ("CHEMBL6", "O=C(Nc1ccc2[nH]ncc2c1)c1ccc(Cl)cc1", 6.9, 2022),
        ("CHEMBL7", "CC(C)Nc1nc(NCc2ccccc2)nc(N)n1", 5.8, 2016),
        ("CHEMBL8", "Cc1nc(C)c(-c2ccc(NC(=O)C3CC3)cc2)s1", 6.1, 2015),
    ]
    df = pd.DataFrame(
        rows, columns=["molecule_chembl_id", "canonical_smiles_std", "pchembl_value", "first_publication_year"]
    )
    df["active"] = df["pchembl_value"] >= 6.5
    p = out_dir / "brd4_curated.parquet"
    df.to_parquet(p, index=False)
    return p


def test_run_extract_writes_canonical_and_manifest(
    tmp_path: Path, tiny_chembl_db: Path, params: Params
) -> None:
    out_dir = tmp_path / "raw"
    out_path = run_extract(out_dir, params, tiny_chembl_db)
    assert out_path.exists() and out_path.name == "brd4_raw.parquet"
    # exactly one run dir, with manifest + git_sha
    run_dirs = list((out_dir / "runs").iterdir())
    assert len(run_dirs) == 1
    rd = run_dirs[0]
    assert (rd / "manifest.json").exists()
    assert (rd / "git_sha.txt").exists()
    manifest = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "extract"
    assert manifest["outputs"]["n_rows"] == 1
    assert manifest["seeds"]["global"] == params.seed


def test_run_split_writes_two_jsons_and_manifest(tmp_path: Path, params: Params) -> None:
    curated_path = _synthetic_curated_parquet(tmp_path / "processed")
    split_dir = tmp_path / "splits"
    paths = run_split(curated_path, split_dir, params)
    assert paths["scaffold"].exists()
    assert paths["time"].exists()
    sc = json.loads(paths["scaffold"].read_text(encoding="utf-8"))
    tm = json.loads(paths["time"].read_text(encoding="utf-8"))
    n = sum(len(sc[k]) for k in ("train", "val", "test"))
    assert n == 8
    assert len(tm["train"]) + len(tm["test"]) == 8
    rd = next((split_dir / "runs").iterdir())
    assert (rd / "manifest.json").exists()


def test_run_featurize_outputs_match_manifest_hashes(
    tmp_path: Path, params: Params
) -> None:
    curated_path = _synthetic_curated_parquet(tmp_path / "processed")
    paths = run_split(curated_path, tmp_path / "splits", params)
    out_dir = tmp_path / "feats"
    feats = run_featurize(curated_path, paths["scaffold"], out_dir, params)
    assert feats["morgan"].exists()
    assert feats["mordred"].exists()
    assert feats["mordred_scaler"].exists()
    assert feats["chemprop"].exists()

    morgan = np.load(feats["morgan"])["X"]
    mordred_X = np.load(feats["mordred"])["X"]
    assert morgan.shape[0] == 8
    assert morgan.shape[1] == params.featurize.morgan.n_bits
    assert mordred_X.shape[0] == 8

    rd = next((out_dir / "runs").iterdir())
    manifest = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
    from brd4kan.utils.hashing import array_sha256

    assert manifest["outputs"]["morgan_array_sha256"] == array_sha256(morgan)
    assert manifest["outputs"]["mordred_array_sha256"] == array_sha256(mordred_X)


def test_run_curate_end_to_end(tmp_path: Path, params: Params) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw = pd.DataFrame(
        [
            {
                "canonical_smiles": "Cc1cc(C)c2c(c1)C(=O)N(Cc1ccc(Cl)cc1)C2=O",
                "pchembl_value": 7.5,
                "first_publication_year": 2018,
                "molecule_chembl_id": "CHEMBL_OK_1",
            },
            {
                "canonical_smiles": "CCO",  # too small
                "pchembl_value": 5.0,
                "first_publication_year": 2018,
                "molecule_chembl_id": "CHEMBL_TINY",
            },
        ]
    )
    raw_path = raw_dir / "brd4_raw.parquet"
    raw.to_parquet(raw_path, index=False)
    out_dir = tmp_path / "processed"
    out = run_curate(raw_path, out_dir, params)
    df = pd.read_parquet(out)
    assert "CHEMBL_OK_1" in df["molecule_chembl_id"].tolist()
    assert "CHEMBL_TINY" not in df["molecule_chembl_id"].tolist()
    rd = next((out_dir / "runs").iterdir())
    m = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
    assert m["outputs"]["n_rows_in"] == 2
    assert m["outputs"]["n_rows_out"] == 1
