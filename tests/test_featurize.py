"""Stage 4 featurizer determinism tests.

PLAN.md gate: same input → byte-identical output.
"""

from __future__ import annotations

import pytest

rdkit = pytest.importorskip("rdkit")
mordred_pkg = pytest.importorskip("mordred")

import numpy as np  # noqa: E402

from brd4kan.features.morgan import morgan_fingerprint, morgan_matrix  # noqa: E402
from brd4kan.features.mordred import MordredFeaturizer  # noqa: E402
from brd4kan.utils.hashing import array_sha256  # noqa: E402


SMILES_SAMPLES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cc1cc(C)c2c(c1)C(=O)N(Cc1ccc(Cl)cc1)C2=O",
    "CN1CCN(c2ccc(N)cc2)CC1",
]


def test_morgan_fingerprint_shape_and_dtype() -> None:
    fp = morgan_fingerprint("CCO", radius=2, n_bits=2048)
    assert fp is not None
    assert fp.shape == (2048,)
    assert fp.dtype == np.uint8
    assert set(np.unique(fp).tolist()).issubset({0, 1})


def test_morgan_invalid_smiles_returns_none() -> None:
    assert morgan_fingerprint("not_a_smiles", radius=2, n_bits=2048) is None


def test_morgan_matrix_byte_identical_across_runs() -> None:
    a, va = morgan_matrix(SMILES_SAMPLES, radius=2, n_bits=2048)
    b, vb = morgan_matrix(SMILES_SAMPLES, radius=2, n_bits=2048)
    assert np.array_equal(a, b)
    assert np.array_equal(va, vb)
    assert array_sha256(a) == array_sha256(b)


def test_morgan_matrix_invalid_row_zeroed_and_flagged() -> None:
    smis = ["CCO", "definitely_not_smiles", "c1ccccc1"]
    mat, valid = morgan_matrix(smis, radius=2, n_bits=2048)
    assert valid.tolist() == [True, False, True]
    assert mat[1].sum() == 0


def test_mordred_fit_transform_byte_identical() -> None:
    f1 = MordredFeaturizer(variance_threshold=0.0, correlation_threshold=0.95)
    f2 = MordredFeaturizer(variance_threshold=0.0, correlation_threshold=0.95)
    a = f1.fit_transform(SMILES_SAMPLES)
    b = f2.fit_transform(SMILES_SAMPLES)
    assert a.shape == b.shape
    assert array_sha256(a) == array_sha256(b)
    assert f1.kept_columns == f2.kept_columns


def test_mordred_transform_after_fit_matches_fit_transform() -> None:
    f = MordredFeaturizer(variance_threshold=0.0, correlation_threshold=0.95)
    a = f.fit_transform(SMILES_SAMPLES)
    b = f.transform(SMILES_SAMPLES)
    assert array_sha256(a) == array_sha256(b)


def test_mordred_save_load_round_trip(tmp_path) -> None:
    f = MordredFeaturizer(variance_threshold=0.0, correlation_threshold=0.95)
    a = f.fit_transform(SMILES_SAMPLES)
    p = tmp_path / "scaler.joblib"
    f.save(p)
    g = MordredFeaturizer.load(p)
    b = g.transform(SMILES_SAMPLES)
    assert array_sha256(a) == array_sha256(b)
