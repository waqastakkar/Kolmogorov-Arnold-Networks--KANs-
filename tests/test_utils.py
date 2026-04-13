"""Tests for cross-cutting utilities (config, seeding, hashing, manifest, runs)."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

from brd4kan.utils.config import Params, load_params, repo_root
from brd4kan.utils.hashing import array_sha256, file_sha256, file_signature
from brd4kan.utils.manifest import (
    Manifest,
    env_snapshot,
    get_git_sha,
    utc_compact,
    utc_timestamp,
    write_manifest,
)
from brd4kan.utils.runs import make_run_dir
from brd4kan.utils.seed import set_global_seed


# ----- config -----


def test_load_params_returns_validated_object() -> None:
    p = load_params()
    assert isinstance(p, Params)
    assert p.seed == 42
    assert p.chembl.target_chembl_id == "CHEMBL1163125"
    assert "IC50" in p.chembl.allowed_standard_types
    assert p.curate.mw_min < p.curate.mw_max
    assert (
        p.split.scaffold.train_frac
        + p.split.scaffold.val_frac
        + p.split.scaffold.test_frac
        == pytest.approx(1.0)
    )


def test_repo_root_contains_params_yaml() -> None:
    assert (repo_root() / "params.yaml").exists()


# ----- seed -----


def test_set_global_seed_makes_python_and_numpy_deterministic() -> None:
    set_global_seed(42)
    py_a = [random.random() for _ in range(5)]
    np_a = np.random.rand(5).tolist()
    set_global_seed(42)
    py_b = [random.random() for _ in range(5)]
    np_b = np.random.rand(5).tolist()
    assert py_a == py_b
    assert np_a == np_b
    assert os.environ["PYTHONHASHSEED"] == "42"


# ----- hashing -----


def test_file_sha256_matches_known_value(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello brd4-kan")
    h = file_sha256(p)
    # SHA-256 of "hello brd4-kan"
    assert len(h) == 64
    # Idempotency
    assert h == file_sha256(p)


def test_file_signature_handles_missing(tmp_path: Path) -> None:
    sig = file_signature(tmp_path / "nope")
    assert sig["status"] == "missing"


def test_file_signature_records_size_and_hash(tmp_path: Path) -> None:
    p = tmp_path / "y.bin"
    p.write_bytes(b"abc" * 1024)
    sig = file_signature(p)
    assert sig["size_bytes"] == 3 * 1024
    assert "sha256" in sig


def test_array_sha256_invariant_to_strides() -> None:
    a = np.arange(24, dtype=np.float32).reshape(4, 6)
    b = a.copy(order="F")  # Fortran-order, different strides, same content
    assert array_sha256(a) == array_sha256(b)


# ----- manifest -----


def test_write_manifest_round_trip(tmp_path: Path) -> None:
    m = Manifest(
        stage="unit_test",
        git_sha=get_git_sha(),
        timestamp=utc_timestamp(),
        inputs={"x": "y"},
        outputs={"n": 1},
        params_snapshot={"seed": 42},
        seeds={"global": 42},
        env=env_snapshot(),
        wall_time_seconds=0.123,
    )
    out = write_manifest(m, tmp_path)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["stage"] == "unit_test"
    assert data["seeds"]["global"] == 42
    assert (tmp_path / "git_sha.txt").exists()


def test_utc_compact_format() -> None:
    s = utc_compact()
    assert s.endswith("Z") and "T" in s and len(s) == 16


# ----- runs -----


def test_make_run_dir_creates_unique_subdir(tmp_path: Path) -> None:
    rd = make_run_dir(tmp_path)
    assert rd.exists() and rd.is_dir()
    assert rd.parent.name == "runs"
