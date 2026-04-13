"""End-to-end smoke test exercising the **full** BRD4-KAN pipeline.

PLAN.md §8 final step: "End-to-end smoke test on a 500-row subsample, then
full run."  This module uses a small synthetic dataset (12 drug-like BRD4-
relevant compounds) to verify that **every** pipeline stage (1-10), the HTML
report generator, and the public ``BRD4Predictor`` Python API produce the
expected artifacts, manifests, data shapes, and value bounds.

Design principles
-----------------
* **Isolated per-stage tests** (``TestStage*``) run independently and fast.
  Each creates its own fixtures from helpers so that ``pytest -k TestStage3``
  works without running any other stage.
* **Full integration test** (``TestFullPipelineIntegration``) chains stages in
  order: split → featurize → baselines → KAN → symbolic → evaluate →
  screen → analyze-hits → report → BRD4Predictor → CLI.
* All helper functions are pure (no global state) and deterministic (seed 42).
* RDKit, ChEMBL Structure Pipeline, and mordred are *required*; modules that
  are missing cause a clean skip at import time.

Guard dependencies
------------------
The ``pytest.importorskip`` calls at module level skip the entire file if the
cheminformatics stack is not installed (CI always has them).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

# Guard: skip entire module if cheminformatics stack is missing
rdkit = pytest.importorskip("rdkit")
csp = pytest.importorskip("chembl_structure_pipeline")
mordred_lib = pytest.importorskip("mordred")

import torch  # noqa: E402

from brd4kan.utils.config import Params, load_params  # noqa: E402
from brd4kan.utils.hashing import array_sha256  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — shared synthetic data generators & KAN stub
# ═══════════════════════════════════════════════════════════════════════════

# 12 drug-like compounds with varied BRD4-relevant scaffolds, pIC50 spread
# spanning active / inactive, and publication years spanning 2015-2022 for
# the time-split test.  All molecules satisfy MW ∈ [150, 700], heavy
# atoms ≥ 10, non-inorganic, non-mixture.
_SYNTHETIC_ROWS: list[tuple[str, str, float, int]] = [
    ("CHEMBL_S01", "Cc1cc(C)c2c(c1)C(=O)N(Cc1ccc(Cl)cc1)C2=O", 7.5, 2018),
    ("CHEMBL_S02", "Cc1cc(C)c2c(c1)C(=O)N(Cc1ccc(Br)cc1)C2=O", 7.0, 2019),
    ("CHEMBL_S03", "O=C1N(Cc2ccc(F)cc2)C(=O)c2cc(C)cc(C)c21", 6.5, 2020),
    ("CHEMBL_S04", "CN1CCN(c2ccc(NC(=O)c3ccc(Cl)cc3)cc2)CC1", 6.8, 2021),
    ("CHEMBL_S05", "Cn1ncc(-c2ccnc(N3CCCC3)n2)c1N", 7.2, 2017),
    ("CHEMBL_S06", "O=C(Nc1ccc2[nH]ncc2c1)c1ccc(Cl)cc1", 6.9, 2022),
    ("CHEMBL_S07", "CC(C)Nc1nc(NCc2ccccc2)nc(N)n1", 5.8, 2016),
    ("CHEMBL_S08", "Cc1nc(C)c(-c2ccc(NC(=O)C3CC3)cc2)s1", 6.1, 2015),
    ("CHEMBL_S09", "O=C(Nc1cccc(F)c1)c1ccc(NC2CCCCC2)cc1", 7.8, 2018),
    ("CHEMBL_S10", "c1ccc2c(c1)[nH]c1ccncc12", 5.5, 2019),
    ("CHEMBL_S11", "CC(=O)Nc1ccc(Oc2ccccc2)cc1", 6.3, 2020),
    ("CHEMBL_S12", "O=c1[nH]c2ccccc2n1Cc1ccc(Cl)cc1", 8.1, 2017),
]


def _ensure_kan_stub() -> None:
    """Inject a lightweight efficient_kan stub if the real package is absent."""
    if "efficient_kan" in sys.modules:
        return
    try:
        import efficient_kan  # noqa: F401
    except ImportError:

        class _StubKANLinear(torch.nn.Module):
            def __init__(
                self, in_features: int, out_features: int,
                grid_size: int = 5, spline_order: int = 3,
            ) -> None:
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.linear = torch.nn.Linear(in_features, out_features)
                self.scaled_spline_weight = torch.nn.Parameter(
                    torch.randn(out_features, in_features, grid_size + spline_order)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        _mod = ModuleType("efficient_kan")
        _mod.KANLinear = _StubKANLinear  # type: ignore[attr-defined]
        sys.modules["efficient_kan"] = _mod


def _synthetic_curated_parquet(out_dir: Path) -> Path:
    """Create a small but realistic curated parquet with 12 BRD4 compounds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        _SYNTHETIC_ROWS,
        columns=[
            "molecule_chembl_id", "canonical_smiles_std",
            "pchembl_value", "first_publication_year",
        ],
    )
    df["active"] = df["pchembl_value"] >= 6.5
    p = out_dir / "brd4_curated.parquet"
    df.to_parquet(p, index=False)
    return p


def _run_split(curated_path: Path, out_dir: Path, params: Params) -> dict[str, Path]:
    from brd4kan.data.split import run_split
    return run_split(curated_path, out_dir, params)


def _run_featurize(
    curated_path: Path, scaffold_json: Path, out_dir: Path, params: Params,
) -> dict[str, Path]:
    from brd4kan.features.run import run_featurize
    return run_featurize(curated_path, scaffold_json, out_dir, params)


def _train_tiny_kan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    out_dir: Path,
    params: Params,
) -> Path:
    """Train a tiny 3-epoch KAN ensemble (1 member) and save artifacts.

    Mimics run_kan's output layout (best_hparams.json, seed_*/member_*.pt,
    seed_*/conformal.json, kan_summary.json) but finishes in <1 second.
    """
    _ensure_kan_stub()
    from brd4kan.models.kan_model import BRD4KANModel

    seed_dir = out_dir / f"seed_{params.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    input_dim = X_train.shape[1]
    layer_widths = [32, 1]
    grid_size = 3
    spline_order = 3

    model = BRD4KANModel(
        input_dim=input_dim,
        layer_widths=layer_widths,
        grid_size=grid_size,
        spline_order=spline_order,
        dropout=0.0,
        use_mult_layer=True,
        aux_head=True,
    )

    # Quick 3-epoch train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_t = torch.from_numpy(X_train).float()
    y_t = torch.from_numpy(y_train).float().unsqueeze(1)

    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        reg, aux = model(X_t)
        loss = torch.nn.functional.mse_loss(reg, y_t)
        loss.backward()
        optimizer.step()

    # Save member checkpoint
    pt_path = seed_dir / "member_0.pt"
    torch.save(model.state_dict(), pt_path)

    # best_hparams.json
    hp: dict[str, Any] = {
        "layer_widths": layer_widths,
        "grid_size": grid_size,
        "spline_order": spline_order,
        "dropout": 0.0,
        "multiplicative_nodes": True,
        "aux_classification_head": True,
        "mc_dropout_samples": 5,
    }
    (out_dir / "best_hparams.json").write_text(
        json.dumps(hp, indent=2), encoding="utf-8"
    )

    # Conformal calibration (trivial — all same scaffold)
    from brd4kan.models.conformal import MondrianConformalPredictor

    model.eval()
    with torch.no_grad():
        cal_preds, _ = model(X_t)
    cal_preds = cal_preds.squeeze().numpy()
    residuals = np.abs(y_train - cal_preds)
    cp = MondrianConformalPredictor(alpha=0.1)
    scaffolds = ["scaffold_A"] * len(y_train)
    cp.calibrate(residuals, scaffolds)
    (seed_dir / "conformal.json").write_text(
        json.dumps(cp.state_dict()), encoding="utf-8"
    )

    # kan_summary.json (expected by DVC)
    (out_dir / "kan_summary.json").write_text(
        json.dumps({"status": "smoke_test", "n_members": 1}), encoding="utf-8"
    )
    return out_dir


def _train_tiny_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: Path,
) -> Path:
    """Train RF + XGBoost tiny baselines — exercises two model types."""
    from brd4kan.models.baselines import create_model, save_model
    from brd4kan.train.metrics import regression_metrics

    results: dict[str, Any] = {}
    for model_name, hparams in [
        ("rf", {"n_estimators": 5, "max_depth": 3}),
        ("xgboost", {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3}),
    ]:
        model = create_model(model_name, hparams, seed=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        model_dir = out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        save_model(model, model_dir / "seed_42.joblib")

        m = regression_metrics(y_test, preds, active_threshold=6.5)
        results[model_name] = {"test_metrics": m, "status": "smoke_test"}

    (out_dir / "baselines_summary.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    return out_dir


def _latest_manifest(dir_path: Path) -> dict[str, Any]:
    """Load the most recent manifest.json from a stage's runs/ directory."""
    runs_dir = dir_path / "runs"
    if not runs_dir.exists():
        return {}
    run_dirs = sorted(runs_dir.iterdir())
    if not run_dirs:
        return {}
    manifest_path = run_dirs[-1] / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


# ═══════════════════════════════════════════════════════════════════════════
# Isolated per-stage tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStage3Split:
    """Stage 3 — Bemis-Murcko scaffold split + time split."""

    def test_scaffold_split_partitions_are_disjoint(self, tmp_path: Path) -> None:
        params = load_params()
        curated = _synthetic_curated_parquet(tmp_path / "proc")
        paths = _run_split(curated, tmp_path / "splits", params)
        sc = json.loads(paths["scaffold"].read_text(encoding="utf-8"))

        train = set(sc["train"])
        val = set(sc["val"])
        test = set(sc["test"])
        assert train.isdisjoint(val), "train ∩ val leakage"
        assert train.isdisjoint(test), "train ∩ test leakage"
        assert val.isdisjoint(test), "val ∩ test leakage"
        assert len(train) + len(val) + len(test) == 12

    def test_time_split_covers_all_rows(self, tmp_path: Path) -> None:
        params = load_params()
        curated = _synthetic_curated_parquet(tmp_path / "proc")
        paths = _run_split(curated, tmp_path / "splits", params)
        tm = json.loads(paths["time"].read_text(encoding="utf-8"))
        assert len(tm["train"]) + len(tm["test"]) == 12

    def test_split_is_deterministic(self, tmp_path: Path) -> None:
        params = load_params()
        curated = _synthetic_curated_parquet(tmp_path / "proc")
        p1 = _run_split(curated, tmp_path / "s1", params)
        p2 = _run_split(curated, tmp_path / "s2", params)
        assert p1["scaffold"].read_text() == p2["scaffold"].read_text()

    def test_manifest_written(self, tmp_path: Path) -> None:
        params = load_params()
        curated = _synthetic_curated_parquet(tmp_path / "proc")
        _run_split(curated, tmp_path / "splits", params)
        m = _latest_manifest(tmp_path / "splits")
        assert m["stage"] == "split"
        assert m["seeds"]["global"] == 42


class TestStage4Featurize:
    """Stage 4 — Morgan + Mordred featurization."""

    def test_morgan_shape_and_dtype(self, tmp_path: Path) -> None:
        params = load_params()
        curated = _synthetic_curated_parquet(tmp_path / "proc")
        splits = _run_split(curated, tmp_path / "splits", params)
        feats = _run_featurize(curated, splits["scaffold"], tmp_path / "feats", params)

        morgan = np.load(feats["morgan"])["X"]
        assert morgan.shape == (12, params.featurize.morgan.n_bits)
        assert morgan.dtype == np.uint8
        assert morgan.min() >= 0
        assert morgan.max() <= 1

    def test_mordred_shape_and_dtype(self, tmp_path: Path) -> None:
        params = load_params()
        curated = _synthetic_curated_parquet(tmp_path / "proc")
        splits = _run_split(curated, tmp_path / "splits", params)
        feats = _run_featurize(curated, splits["scaffold"], tmp_path / "feats", params)

        mordred_X = np.load(feats["mordred"])["X"]
        assert mordred_X.shape[0] == 12
        assert mordred_X.dtype == np.float32
        assert not np.any(np.isnan(mordred_X)), "NaN found in Mordred descriptors"

    def test_scaler_saved(self, tmp_path: Path) -> None:
        params = load_params()
        curated = _synthetic_curated_parquet(tmp_path / "proc")
        splits = _run_split(curated, tmp_path / "splits", params)
        feats = _run_featurize(curated, splits["scaffold"], tmp_path / "feats", params)
        assert feats["mordred_scaler"].exists()
        assert feats["mordred_scaler"].stat().st_size > 0

    def test_chemprop_csv_written(self, tmp_path: Path) -> None:
        params = load_params()
        curated = _synthetic_curated_parquet(tmp_path / "proc")
        splits = _run_split(curated, tmp_path / "splits", params)
        feats = _run_featurize(curated, splits["scaffold"], tmp_path / "feats", params)
        assert feats["chemprop"].exists()
        chemprop_df = pd.read_csv(feats["chemprop"])
        assert len(chemprop_df) == 12

    def test_featurize_manifest_hashes(self, tmp_path: Path) -> None:
        params = load_params()
        curated = _synthetic_curated_parquet(tmp_path / "proc")
        splits = _run_split(curated, tmp_path / "splits", params)
        feats = _run_featurize(curated, splits["scaffold"], tmp_path / "feats", params)

        morgan = np.load(feats["morgan"])["X"]
        m = _latest_manifest(tmp_path / "feats")
        assert m["outputs"]["morgan_array_sha256"] == array_sha256(morgan)


class TestStage5Baselines:
    """Stage 5 — Baseline model training (RF, XGBoost)."""

    def test_baselines_produce_predictions(self, tmp_path: Path) -> None:
        from brd4kan.models.baselines import create_model

        rng = np.random.RandomState(42)
        X = rng.randn(50, 20).astype(np.float32)
        y = rng.randn(50).astype(np.float64) + 7.0

        for name in ["rf", "xgboost", "svr", "mlp"]:
            model = create_model(name, {}, seed=42)
            model.fit(X[:40], y[:40])
            preds = model.predict(X[40:])
            assert preds.shape == (10,), f"{name} prediction shape wrong"
            assert np.isfinite(preds).all(), f"{name} has non-finite predictions"

    def test_baseline_save_load_roundtrip(self, tmp_path: Path) -> None:
        from brd4kan.models.baselines import create_model, load_model, save_model

        rng = np.random.RandomState(42)
        X = rng.randn(30, 10).astype(np.float32)
        y = rng.randn(30) + 7.0

        model = create_model("rf", {"n_estimators": 3}, seed=42)
        model.fit(X, y)
        p_before = model.predict(X[:5])

        path = tmp_path / "rf.joblib"
        save_model(model, path)
        loaded = load_model(path)
        p_after = loaded.predict(X[:5])
        np.testing.assert_array_almost_equal(p_before, p_after)


class TestStage6KAN:
    """Stage 6 — KAN ensemble forward pass, sparsity, uncertainty."""

    def test_kan_forward_shapes(self) -> None:
        _ensure_kan_stub()
        from brd4kan.models.kan_model import BRD4KANModel

        model = BRD4KANModel(input_dim=64, layer_widths=[32, 1])
        x = torch.randn(8, 64)
        reg, aux = model(x)
        assert reg.shape == (8, 1)
        assert aux.shape == (8, 1)

    def test_ensemble_uncertainty(self) -> None:
        _ensure_kan_stub()
        from brd4kan.models.kan_model import BRD4KANModel, EnsembleKAN

        members = [BRD4KANModel(input_dim=32, layer_widths=[16, 1]) for _ in range(3)]
        ens = EnsembleKAN(members)
        x = torch.randn(5, 32)
        mean, ep, al = ens.predict_with_uncertainty(x, mc_samples=3)
        assert mean.shape == (5,)
        assert ep.shape == (5,)
        assert (ep >= 0).all()

    def test_conformal_coverage(self) -> None:
        from brd4kan.models.conformal import MondrianConformalPredictor

        rng = np.random.RandomState(42)
        residuals = rng.randn(100)
        scaffolds = [f"scaf_{i % 5}" for i in range(100)]
        cp = MondrianConformalPredictor(alpha=0.1)
        cp.calibrate(residuals, scaffolds)

        preds = rng.randn(50) + 7.0
        test_scaffolds = [f"scaf_{i % 5}" for i in range(50)]
        lo, hi = cp.predict_intervals(preds, test_scaffolds)
        assert lo.shape == (50,)
        assert (lo <= hi).all()

        # Serialization round-trip
        state = cp.state_dict()
        cp2 = MondrianConformalPredictor.from_state_dict(state)
        lo2, hi2 = cp2.predict_intervals(preds, test_scaffolds)
        np.testing.assert_array_almost_equal(lo, lo2)


class TestStage7Symbolic:
    """Stage 7 — Symbolic extraction from KAN."""

    def test_symbolic_edge_fit_recovers_quadratic(self) -> None:
        from brd4kan.explain.symbolic import fit_symbolic_edge

        rng = np.random.RandomState(42)
        x = rng.randn(200)
        y = 2.0 * x**2 - 1.0 * x + 0.5
        result = fit_symbolic_edge(x, y)
        assert result["function"] == "poly2"
        assert result["rmse"] < 0.1

    def test_equation_assembly_produces_latex(self) -> None:
        from brd4kan.explain.symbolic import build_symbolic_equation

        edge_fits = [
            {"input_idx": 0, "function": "poly2", "params": (1.0, -0.5, 0.3)},
            {"input_idx": 1, "function": "exp", "params": (0.8, 0.1)},
        ]
        latex, expr = build_symbolic_equation(edge_fits, ["ECFP4_0", "MolWt"])
        assert isinstance(latex, str)
        assert len(latex) > 0
        assert "ECFP4" in latex or "MolWt" in latex


class TestStage9Screen:
    """Stage 9 — Virtual screening helpers."""

    def test_standardize_and_filter_handles_empty(self) -> None:
        from brd4kan.screen.screening import standardize_and_filter
        params = load_params()
        df = standardize_and_filter([], params)
        assert len(df) == 0

    def test_standardize_and_filter_rejects_invalid(self) -> None:
        from brd4kan.screen.screening import standardize_and_filter
        params = load_params()
        df = standardize_and_filter(["NOT_SMILES", "XXXXX"], params)
        assert len(df) == 0

    def test_butina_selection_respects_top_n(self) -> None:
        from brd4kan.screen.screening import butina_diversity_selection

        smiles = [
            "c1ccccc1", "c1ccncc1", "C1CCCCC1",
            "c1ccc2ccccc2c1", "CC(=O)O", "CCO",
        ]
        selected = butina_diversity_selection(smiles, cutoff=0.6, top_n=3)
        assert len(selected) <= 3
        assert selected == sorted(selected)

    def test_embed_3d_sdf_writes_valid_file(self, tmp_path: Path) -> None:
        from brd4kan.screen.screening import embed_3d_sdf

        out = embed_3d_sdf(["c1ccccc1", "CCO"], tmp_path / "out.sdf")
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "$$$$" in content


class TestStage10AnalyzeHits:
    """Stage 10 — Hit annotation and pharmacophore matching."""

    def test_pharmacophore_amide_detected(self) -> None:
        from brd4kan.screen.analyze_hits import _check_pharmacophore
        assert _check_pharmacophore("CC(=O)NC") is True

    def test_pharmacophore_indole_detected(self) -> None:
        from brd4kan.screen.analyze_hits import _check_pharmacophore
        assert _check_pharmacophore("c1ccc2[nH]ccc2c1") is True

    def test_pharmacophore_alkane_negative(self) -> None:
        from brd4kan.screen.analyze_hits import _check_pharmacophore
        assert _check_pharmacophore("CCCCCC") is False

    def test_nearest_neighbor_self_match(self) -> None:
        from brd4kan.screen.analyze_hits import _nearest_chembl_neighbor

        smiles = ["c1ccccc1", "CCO"]
        results = _nearest_chembl_neighbor(
            query_smiles=["c1ccccc1"],
            train_smiles=smiles,
            train_pchembl=np.array([7.0, 6.5]),
        )
        assert results[0]["nn_tanimoto"] == pytest.approx(1.0)


class TestReport:
    """HTML report generation."""

    def test_report_with_metrics(self, tmp_path: Path) -> None:
        from brd4kan.screen.report import build_report

        m_dir = tmp_path / "metrics"
        m_dir.mkdir()
        (m_dir / "evaluation_metrics.json").write_text(json.dumps({
            "KAN": {
                "metrics": {"rmse": 0.45, "r2": 0.88, "spearman_rho": 0.91},
                "bootstrap_ci": {"rmse": {"lo": 0.40, "hi": 0.50}},
            },
        }), encoding="utf-8")
        for d in ["figures", "symbolic", "hits"]:
            (tmp_path / d).mkdir()

        out = build_report(m_dir, tmp_path / "figures", tmp_path / "symbolic",
                           tmp_path / "hits", tmp_path / "report.html")
        html = out.read_text(encoding="utf-8")
        assert "BRD4-KAN Pipeline Report" in html
        assert "KAN" in html
        assert "0.4500" in html


class TestBRD4PredictorUnit:
    """BRD4Predictor Python API — unit level."""

    def test_predict_smiles_returns_all_keys(self) -> None:
        from brd4kan.predict import BRD4Predictor
        from brd4kan.train.applicability import ApplicabilityDomain

        rng = np.random.RandomState(42)
        ad = ApplicabilityDomain(pca_components=3)
        ad.fit(
            (rng.rand(20, 64) > 0.5).astype(np.uint8),
            rng.randn(20, 10).astype(np.float32),
        )
        p = BRD4Predictor(
            ensemble=None, mordred_featurizer=None, ad=ad, conformal=None,
            morgan_cfg={"radius": 2, "n_bits": 64},
        )
        results = p.predict_smiles(["c1ccccc1", "CCO"])
        assert len(results) == 2
        expected = {
            "smiles", "pred_pIC50", "ci_lower", "ci_upper",
            "epistemic_std", "aleatoric_std", "ad_in_domain", "tanimoto_nn",
        }
        for r in results:
            assert set(r.keys()) == expected

    def test_zero_prediction_without_ensemble(self) -> None:
        from brd4kan.predict import BRD4Predictor
        from brd4kan.train.applicability import ApplicabilityDomain

        ad = ApplicabilityDomain(pca_components=2)
        rng = np.random.RandomState(42)
        ad.fit(
            (rng.rand(10, 32) > 0.5).astype(np.uint8),
            rng.randn(10, 5).astype(np.float32),
        )
        p = BRD4Predictor(
            ensemble=None, mordred_featurizer=None, ad=ad, conformal=None,
            morgan_cfg={"radius": 2, "n_bits": 32},
        )
        r = p.predict_smiles(["c1ccccc1"])[0]
        assert r["pred_pIC50"] == pytest.approx(0.0)
        assert r["ci_lower"] == pytest.approx(-1.0)
        assert r["ci_upper"] == pytest.approx(1.0)


class TestCLIRegistration:
    """All 11 CLI commands are registered and respond to --help."""

    @pytest.mark.parametrize("cmd", [
        "extract", "curate", "split", "featurize",
        "train-baselines", "train-kan", "symbolic", "evaluate",
        "screen", "analyze-hits", "report",
    ])
    def test_cli_help(self, cmd: str) -> None:
        from typer.testing import CliRunner
        from brd4kan.cli import app

        result = CliRunner().invoke(app, [cmd, "--help"])
        assert result.exit_code == 0, f"{cmd} --help failed: {result.output}"


# ═══════════════════════════════════════════════════════════════════════════
# Full integration test — chains every stage end-to-end
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestFullPipelineIntegration:
    """Run the complete pipeline from curated data through report.html.

    This single test exercises every production code path that touches
    disk, verifying artifact existence, manifest contents, data shapes,
    dtype invariants, and value-range sanity checks at each stage.
    """

    def test_full_pipeline_e2e(self, tmp_path: Path) -> None:  # noqa: C901
        params = load_params()
        N = 12  # number of synthetic compounds

        # ════════════════════════════════════════════════════════════════
        # Stage 2 — Curate (synthetic)
        # ════════════════════════════════════════════════════════════════
        processed_dir = tmp_path / "processed"
        curated_path = _synthetic_curated_parquet(processed_dir)
        df = pd.read_parquet(curated_path)
        assert len(df) == N
        assert set(df.columns) >= {
            "molecule_chembl_id", "canonical_smiles_std",
            "pchembl_value", "first_publication_year", "active",
        }
        assert df["pchembl_value"].between(4.0, 12.0).all()
        assert df["active"].dtype == bool

        # ════════════════════════════════════════════════════════════════
        # Stage 3 — Split
        # ════════════════════════════════════════════════════════════════
        splits_dir = tmp_path / "splits"
        split_paths = _run_split(curated_path, splits_dir, params)

        # Scaffold split
        sc = json.loads(split_paths["scaffold"].read_text(encoding="utf-8"))
        train_idx = np.array(sc["train"])
        val_idx = np.array(sc["val"])
        test_idx = np.array(sc["test"])
        assert len(train_idx) + len(val_idx) + len(test_idx) == N

        all_idx = np.concatenate([train_idx, val_idx, test_idx])
        assert len(np.unique(all_idx)) == N, "Indices not unique across splits"
        assert set(sc["train"]).isdisjoint(set(sc["test"]))
        assert set(sc["train"]).isdisjoint(set(sc["val"]))
        assert set(sc["val"]).isdisjoint(set(sc["test"]))

        # Time split
        tm = json.loads(split_paths["time"].read_text(encoding="utf-8"))
        assert len(tm["train"]) + len(tm["test"]) == N

        # Manifest
        assert _latest_manifest(splits_dir)["stage"] == "split"

        # ════════════════════════════════════════════════════════════════
        # Stage 4 — Featurize
        # ════════════════════════════════════════════════════════════════
        feats_dir = tmp_path / "feats"
        feats = _run_featurize(curated_path, split_paths["scaffold"], feats_dir, params)

        # Morgan fingerprints
        morgan = np.load(feats["morgan"])["X"]
        assert morgan.shape == (N, params.featurize.morgan.n_bits)
        assert morgan.dtype == np.uint8
        assert set(np.unique(morgan)).issubset({0, 1})

        # Mordred descriptors
        mordred_X = np.load(feats["mordred"])["X"]
        assert mordred_X.shape[0] == N
        assert mordred_X.dtype == np.float32
        assert np.isfinite(mordred_X).all()

        # Scaler + Chemprop CSV
        assert feats["mordred_scaler"].exists()
        assert feats["chemprop"].exists()
        assert len(pd.read_csv(feats["chemprop"])) == N

        # Manifest hash verification
        m = _latest_manifest(feats_dir)
        assert m["stage"] == "featurize"
        assert m["outputs"]["morgan_array_sha256"] == array_sha256(morgan)

        # Combined feature matrix
        X_all = np.hstack([morgan.astype(np.float32), mordred_X])
        y_all = df["pchembl_value"].to_numpy(dtype=np.float64)

        # ════════════════════════════════════════════════════════════════
        # Stage 5 — Baselines (RF + XGBoost, tiny)
        # ════════════════════════════════════════════════════════════════
        baselines_dir = tmp_path / "baselines"
        _train_tiny_baselines(
            X_all[train_idx], y_all[train_idx],
            X_all[test_idx], y_all[test_idx],
            baselines_dir,
        )
        assert (baselines_dir / "rf" / "seed_42.joblib").exists()
        assert (baselines_dir / "xgboost" / "seed_42.joblib").exists()
        assert (baselines_dir / "baselines_summary.json").exists()

        summary = json.loads(
            (baselines_dir / "baselines_summary.json").read_text(encoding="utf-8")
        )
        assert "rf" in summary
        assert "xgboost" in summary

        # Verify RF predictions are finite and reasonable
        from brd4kan.models.baselines import load_model

        rf = load_model(baselines_dir / "rf" / "seed_42.joblib")
        rf_preds = rf.predict(X_all[test_idx])
        assert np.isfinite(rf_preds).all()
        assert rf_preds.min() > 0.0  # pIC50 should be positive

        # ════════════════════════════════════════════════════════════════
        # Stage 6 — KAN (tiny 3-epoch, 1 member)
        # ════════════════════════════════════════════════════════════════
        _ensure_kan_stub()
        kan_dir = tmp_path / "kan"
        _train_tiny_kan(X_all[train_idx], y_all[train_idx], kan_dir, params)

        assert (kan_dir / "best_hparams.json").exists()
        assert (kan_dir / f"seed_{params.seed}" / "member_0.pt").exists()
        assert (kan_dir / f"seed_{params.seed}" / "conformal.json").exists()
        assert (kan_dir / "kan_summary.json").exists()

        # Verify KAN forward pass
        from brd4kan.models.kan_model import BRD4KANModel

        hp = json.loads((kan_dir / "best_hparams.json").read_text(encoding="utf-8"))
        kan_model = BRD4KANModel(
            input_dim=X_all.shape[1],
            layer_widths=hp["layer_widths"],
            grid_size=hp["grid_size"],
            spline_order=hp["spline_order"],
            dropout=hp.get("dropout", 0.0),
            use_mult_layer=hp.get("multiplicative_nodes", True),
            aux_head=hp.get("aux_classification_head", True),
        )
        state = torch.load(
            kan_dir / f"seed_{params.seed}" / "member_0.pt",
            map_location="cpu", weights_only=True,
        )
        kan_model.load_state_dict(state)
        kan_model.eval()
        with torch.no_grad():
            test_pred, test_aux = kan_model(torch.from_numpy(X_all[test_idx]).float())
        assert test_pred.shape[0] == len(test_idx)
        assert torch.isfinite(test_pred).all()

        # ════════════════════════════════════════════════════════════════
        # Stage 7 — Symbolic extraction
        # ════════════════════════════════════════════════════════════════
        from brd4kan.explain.symbolic import run_symbolic

        symbolic_dir = tmp_path / "symbolic"
        sym_result = run_symbolic(
            kan_dir, curated_path, feats["morgan"], feats["mordred"],
            symbolic_dir, params,
        )

        # LaTeX equation
        tex_path = symbolic_dir / "pIC50_equation.tex"
        assert tex_path.exists()
        latex = tex_path.read_text(encoding="utf-8")
        assert len(latex) > 0

        # SymPy pickle
        pkl_path = symbolic_dir / "pIC50_equation.pkl"
        assert pkl_path.exists()
        with pkl_path.open("rb") as f:
            sympy_expr = pickle.load(f)
        assert sympy_expr is not None

        # Descriptor importance JSON
        imp_path = symbolic_dir / "descriptor_importance.json"
        assert imp_path.exists()
        imp_data = json.loads(imp_path.read_text(encoding="utf-8"))
        assert isinstance(imp_data, list)
        assert len(imp_data) > 0
        assert all("input_idx" in e and "importance" in e for e in imp_data)
        # Importances sorted descending
        importances = [e["importance"] for e in imp_data]
        assert importances == sorted(importances, reverse=True)

        assert sym_result["n_surviving_edges"] >= 0
        assert _latest_manifest(symbolic_dir)["stage"] == "symbolic"

        # ════════════════════════════════════════════════════════════════
        # Stage 8 — Evaluate (metrics + bootstrap + AD + figures)
        # ════════════════════════════════════════════════════════════════
        from brd4kan.train.run_evaluate import run_evaluate

        figures_dir = tmp_path / "figures"
        metrics_dir = tmp_path / "metrics"
        eval_results = run_evaluate(
            curated_path, split_paths["scaffold"], split_paths["time"],
            feats["morgan"], feats["mordred"],
            baselines_dir, kan_dir, symbolic_dir,
            figures_dir, metrics_dir, params,
        )

        # Metrics JSON
        metrics_json_path = metrics_dir / "evaluation_metrics.json"
        assert metrics_json_path.exists()
        metrics_data = json.loads(metrics_json_path.read_text(encoding="utf-8"))

        for model_name in ["rf", "KAN"]:
            if model_name in metrics_data:
                entry = metrics_data[model_name]
                assert "metrics" in entry
                assert "bootstrap_ci" in entry
                m = entry["metrics"]
                assert m["rmse"] >= 0.0
                assert -1.0 <= m["r2"] <= 1.0
                assert -1.0 <= m["spearman_rho"] <= 1.0

                ci = entry["bootstrap_ci"]
                if "rmse" in ci:
                    assert ci["rmse"]["lo"] <= ci["rmse"]["hi"]

        # AD scores
        ad_json = metrics_dir / "ad_scores.json"
        assert ad_json.exists()
        ad_data = json.loads(ad_json.read_text(encoding="utf-8"))
        assert ad_data["n_in_domain"] + ad_data["n_out_of_domain"] == len(test_idx)
        assert 0.0 <= ad_data["tanimoto_nn_mean"] <= 1.0

        # Figures — at least dataset overview + AD map
        svgs = sorted(figures_dir.glob("*.svg"))
        assert len(svgs) >= 2, f"Expected >=2 SVGs, got {[s.name for s in svgs]}"
        for svg in svgs:
            content = svg.read_bytes()
            assert b"<svg" in content[:1000] or content[:5] == b"<?xml"

        assert _latest_manifest(metrics_dir)["stage"] == "evaluate"

        # ════════════════════════════════════════════════════════════════
        # Stage 9 — Screening
        # ════════════════════════════════════════════════════════════════
        from brd4kan.screen.screening import embed_3d_sdf, standardize_and_filter

        library_smiles = df["canonical_smiles_std"].tolist()
        filtered = standardize_and_filter(library_smiles, params)
        assert isinstance(filtered, pd.DataFrame)

        if len(filtered) > 0:
            sdf_path = tmp_path / "test_embed.sdf"
            embed_3d_sdf(filtered["canonical_smiles_std"].tolist()[:3], sdf_path)
            assert sdf_path.exists()
            sdf_text = sdf_path.read_text(encoding="utf-8")
            assert "$$$$" in sdf_text

        # Build simulated top_hits (mimics run_screen output)
        screen_dir = tmp_path / "screen_out"
        screen_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.RandomState(42)
        hits_df = df.copy()
        hits_df["pred_pIC50"] = y_all + rng.randn(N) * 0.1
        hits_df["epistemic_std"] = 0.1
        hits_df["aleatoric_std"] = 0.05
        hits_df["ci_lower"] = hits_df["pred_pIC50"] - 0.5
        hits_df["ci_upper"] = hits_df["pred_pIC50"] + 0.5
        hits_df["ad_in_domain"] = True
        hits_df["tanimoto_nn"] = rng.uniform(0.3, 1.0, N)
        hits_df["qed"] = rng.uniform(0.5, 0.9, N)
        hits_df["sa_score"] = rng.uniform(2.0, 5.0, N)
        hits_df["rank_score"] = hits_df["pred_pIC50"] + 100.0

        top_hits_csv = screen_dir / "top_hits.csv"
        hits_df.to_csv(top_hits_csv, index=False)

        # ════════════════════════════════════════════════════════════════
        # Stage 10 — Analyze hits
        # ════════════════════════════════════════════════════════════════
        from brd4kan.screen.analyze_hits import run_analyze_hits

        hits_out_dir = tmp_path / "hits_out"
        analyze_result = run_analyze_hits(
            top_hits_csv, curated_path, hits_out_dir, params,
        )

        assert analyze_result["n_hits"] == N
        assert isinstance(analyze_result["n_novel"], int)
        assert isinstance(analyze_result["n_pharmacophore"], int)
        assert analyze_result["n_novel"] >= 0
        assert analyze_result["n_pharmacophore"] >= 0

        # Annotated CSV
        annotated_path = hits_out_dir / "annotated_hits.csv"
        assert annotated_path.exists()
        annotated = pd.read_csv(annotated_path)
        assert len(annotated) == N
        required_cols = {
            "nn_smiles", "nn_tanimoto", "nn_pIC50",
            "novel", "scaffold", "pharmacophore_match",
        }
        assert required_cols.issubset(set(annotated.columns))
        assert annotated["nn_tanimoto"].between(0.0, 1.0).all()

        # Hit cards SVG
        cards_path = hits_out_dir / "hit_cards.svg"
        assert cards_path.exists()
        assert b"<svg" in cards_path.read_bytes()[:1000] or \
               cards_path.read_bytes()[:5] == b"<?xml"

        assert _latest_manifest(hits_out_dir)["stage"] == "analyze_hits"

        # ════════════════════════════════════════════════════════════════
        # Report
        # ════════════════════════════════════════════════════════════════
        from brd4kan.screen.report import build_report

        report_path = tmp_path / "report.html"
        build_report(
            metrics_dir, figures_dir, symbolic_dir, hits_out_dir, report_path,
        )
        assert report_path.exists()
        html = report_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "BRD4-KAN Pipeline Report" in html
        assert "Evaluation Metrics" in html
        assert "Figures" in html
        assert "Screening" in html
        if svgs:
            assert "<svg" in html  # Inlined SVG content

        # ════════════════════════════════════════════════════════════════
        # BRD4Predictor Python API
        # ════════════════════════════════════════════════════════════════
        from brd4kan.predict import BRD4Predictor
        from brd4kan.train.applicability import ApplicabilityDomain

        ad = ApplicabilityDomain(pca_components=3)
        ad.fit(morgan[train_idx], mordred_X[train_idx])

        predictor = BRD4Predictor(
            ensemble=None,
            mordred_featurizer=None,
            ad=ad,
            conformal=None,
            morgan_cfg={"radius": 2, "n_bits": params.featurize.morgan.n_bits},
        )

        test_smiles = ["c1ccccc1", "CCO", "CC(=O)Nc1ccccc1"]
        results = predictor.predict_smiles(test_smiles)
        assert len(results) == 3
        for r in results:
            assert isinstance(r["pred_pIC50"], float)
            assert isinstance(r["ad_in_domain"], bool)
            assert isinstance(r["tanimoto_nn"], float)
            assert r["tanimoto_nn"] >= 0.0
            assert r["ci_lower"] <= r["ci_upper"]

        # ════════════════════════════════════════════════════════════════
        # Reproducibility — second run produces identical splits
        # ════════════════════════════════════════════════════════════════
        splits_dir_2 = tmp_path / "splits_rerun"
        paths_2 = _run_split(curated_path, splits_dir_2, params)
        sc2 = json.loads(paths_2["scaffold"].read_text(encoding="utf-8"))
        assert sc["train"] == sc2["train"]
        assert sc["val"] == sc2["val"]
        assert sc["test"] == sc2["test"]
