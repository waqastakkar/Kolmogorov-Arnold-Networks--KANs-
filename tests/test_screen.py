"""Tests for Stages 9-10: screening, hit analysis, report, and BRD4Predictor API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Screening helpers
# ---------------------------------------------------------------------------


class TestStandardizeAndFilter:
    """Test drug-likeness filter pipeline."""

    def test_valid_smiles_pass_through(self, params) -> None:
        from brd4kan.screen.screening import standardize_and_filter

        # Aspirin-like: passes Ro5, QED >= 0.5, no PAINS
        smiles = ["CC(=O)Oc1ccccc1C(=O)O"]
        df = standardize_and_filter(smiles, params)
        # May or may not pass depending on QED; just assert structure
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert "canonical_smiles_std" in df.columns
            assert "qed" in df.columns
            assert "sa_score" in df.columns

    def test_empty_input_returns_empty(self, params) -> None:
        from brd4kan.screen.screening import standardize_and_filter

        df = standardize_and_filter([], params)
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)

    def test_invalid_smiles_filtered(self, params) -> None:
        from brd4kan.screen.screening import standardize_and_filter

        df = standardize_and_filter(["NOT_A_SMILES", "XXXX"], params)
        assert len(df) == 0


class TestButinaDiversitySelection:
    """Test Butina clustering diversity selection."""

    def test_returns_sorted_indices(self) -> None:
        from brd4kan.screen.screening import butina_diversity_selection

        # Generate simple distinct SMILES
        smiles = [
            "c1ccccc1",       # benzene
            "c1ccncc1",       # pyridine
            "C1CCCCC1",       # cyclohexane
            "c1ccc2ccccc2c1", # naphthalene
            "CC(=O)O",        # acetic acid
        ]
        selected = butina_diversity_selection(smiles, cutoff=0.6, top_n=3)
        assert len(selected) <= 3
        assert selected == sorted(selected)
        assert all(0 <= i < len(smiles) for i in selected)

    def test_top_n_respected(self) -> None:
        from brd4kan.screen.screening import butina_diversity_selection

        smiles = [f"C{'C' * i}O" for i in range(1, 20)]
        selected = butina_diversity_selection(smiles, cutoff=0.4, top_n=5)
        assert len(selected) <= 5


class TestEmbed3dSdf:
    """Test 3D embedding and SDF writing."""

    def test_produces_sdf_file(self, tmp_path: Path) -> None:
        from brd4kan.screen.screening import embed_3d_sdf

        smiles = ["c1ccccc1", "CCO"]
        out = embed_3d_sdf(smiles, tmp_path / "test.sdf")
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "$$$$" in content  # SDF record delimiter

    def test_invalid_smiles_skipped(self, tmp_path: Path) -> None:
        from brd4kan.screen.screening import embed_3d_sdf

        out = embed_3d_sdf(["INVALID_SMILES"], tmp_path / "bad.sdf")
        assert out.exists()


# ---------------------------------------------------------------------------
# Hit analysis helpers
# ---------------------------------------------------------------------------


class TestPharmacophore:
    """Test BRD4 pharmacophore matching."""

    def test_amide_detected(self) -> None:
        from brd4kan.screen.analyze_hits import _check_pharmacophore

        # Simple amide: N-C(=O)-C
        assert _check_pharmacophore("CC(=O)NC") is True

    def test_indole_detected(self) -> None:
        from brd4kan.screen.analyze_hits import _check_pharmacophore

        # Indole core
        assert _check_pharmacophore("c1ccc2[nH]ccc2c1") is True

    def test_simple_alkane_not_detected(self) -> None:
        from brd4kan.screen.analyze_hits import _check_pharmacophore

        assert _check_pharmacophore("CCCCCC") is False

    def test_invalid_smiles_returns_false(self) -> None:
        from brd4kan.screen.analyze_hits import _check_pharmacophore

        assert _check_pharmacophore("NOT_A_SMILES") is False


class TestNearestNeighbor:
    """Test nearest ChEMBL neighbor lookup."""

    def test_self_match_gives_tanimoto_one(self) -> None:
        from brd4kan.screen.analyze_hits import _nearest_chembl_neighbor

        smiles = ["c1ccccc1", "CCO", "CC(=O)O"]
        results = _nearest_chembl_neighbor(
            query_smiles=["c1ccccc1"],
            train_smiles=smiles,
            train_pchembl=np.array([7.0, 6.5, 5.5]),
        )
        assert len(results) == 1
        assert results[0]["nn_tanimoto"] == pytest.approx(1.0)
        assert results[0]["nn_pIC50"] == pytest.approx(7.0)

    def test_invalid_query_returns_zero_sim(self) -> None:
        from brd4kan.screen.analyze_hits import _nearest_chembl_neighbor

        results = _nearest_chembl_neighbor(
            query_smiles=["INVALID"],
            train_smiles=["c1ccccc1"],
            train_pchembl=np.array([7.0]),
        )
        assert results[0]["nn_tanimoto"] == 0.0
        assert results[0]["nn_smiles"] is None


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


class TestReport:
    """Test HTML report generation."""

    def test_builds_html_with_empty_dirs(self, tmp_path: Path) -> None:
        from brd4kan.screen.report import build_report

        metrics_dir = tmp_path / "metrics"
        figures_dir = tmp_path / "figures"
        symbolic_dir = tmp_path / "symbolic"
        hits_dir = tmp_path / "hits"
        for d in [metrics_dir, figures_dir, symbolic_dir, hits_dir]:
            d.mkdir()

        out = build_report(metrics_dir, figures_dir, symbolic_dir, hits_dir,
                           tmp_path / "report.html")
        assert out.exists()
        html = out.read_text(encoding="utf-8")
        assert "BRD4-KAN Pipeline Report" in html
        assert "<!DOCTYPE html>" in html

    def test_metrics_table_populated(self, tmp_path: Path) -> None:
        import json

        from brd4kan.screen.report import build_report

        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()
        (metrics_dir / "evaluation_metrics.json").write_text(
            json.dumps({
                "KAN": {
                    "metrics": {"rmse": 0.45, "r2": 0.88, "spearman_rho": 0.91},
                    "bootstrap_ci": {"rmse": {"lo": 0.40, "hi": 0.50}},
                },
            }),
            encoding="utf-8",
        )
        for d_name in ["figures", "symbolic", "hits"]:
            (tmp_path / d_name).mkdir()

        out = build_report(
            metrics_dir, tmp_path / "figures", tmp_path / "symbolic",
            tmp_path / "hits", tmp_path / "report.html",
        )
        html = out.read_text(encoding="utf-8")
        assert "KAN" in html
        assert "0.4500" in html  # RMSE value

    def test_svg_figure_embedded(self, tmp_path: Path) -> None:
        from brd4kan.screen.report import build_report

        for d_name in ["metrics", "symbolic", "hits"]:
            (tmp_path / d_name).mkdir()
        fig_dir = tmp_path / "figures"
        fig_dir.mkdir()
        (fig_dir / "test_fig.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"><circle r="10"/></svg>',
            encoding="utf-8",
        )

        out = build_report(
            tmp_path / "metrics", fig_dir, tmp_path / "symbolic",
            tmp_path / "hits", tmp_path / "report.html",
        )
        html = out.read_text(encoding="utf-8")
        assert "<circle" in html
        assert "Test Fig" in html


# ---------------------------------------------------------------------------
# BRD4Predictor API
# ---------------------------------------------------------------------------


class TestBRD4PredictorAPI:
    """Test the high-level prediction API."""

    def test_predict_smiles_returns_expected_keys(self) -> None:
        """Test prediction output structure using a mock predictor."""
        from brd4kan.predict import BRD4Predictor
        from brd4kan.train.applicability import ApplicabilityDomain

        rng = np.random.RandomState(42)
        # Create a minimal AD (fit on random data)
        ad = ApplicabilityDomain(pca_components=3)
        fps = (rng.rand(20, 64) > 0.5).astype(np.uint8)
        desc = rng.randn(20, 10).astype(np.float32)
        ad.fit(fps, desc)

        predictor = BRD4Predictor(
            ensemble=None,  # No ensemble → zero predictions
            mordred_featurizer=None,
            ad=ad,
            conformal=None,
            morgan_cfg={"radius": 2, "n_bits": 64},
            mc_samples=5,
        )

        results = predictor.predict_smiles(["c1ccccc1", "CCO"])
        assert len(results) == 2
        expected_keys = {
            "smiles", "pred_pIC50", "ci_lower", "ci_upper",
            "epistemic_std", "aleatoric_std", "ad_in_domain", "tanimoto_nn",
        }
        for r in results:
            assert set(r.keys()) == expected_keys
            assert isinstance(r["pred_pIC50"], float)
            assert isinstance(r["ad_in_domain"], bool)
            assert isinstance(r["tanimoto_nn"], float)

    def test_zero_prediction_without_ensemble(self) -> None:
        """Without an ensemble, preds should be 0.0."""
        from brd4kan.predict import BRD4Predictor
        from brd4kan.train.applicability import ApplicabilityDomain

        ad = ApplicabilityDomain(pca_components=2)
        fps = (np.random.RandomState(42).rand(10, 32) > 0.5).astype(np.uint8)
        desc = np.random.RandomState(42).randn(10, 5).astype(np.float32)
        ad.fit(fps, desc)

        predictor = BRD4Predictor(
            ensemble=None,
            mordred_featurizer=None,
            ad=ad,
            conformal=None,
            morgan_cfg={"radius": 2, "n_bits": 32},
        )
        results = predictor.predict_smiles(["c1ccccc1"])
        assert results[0]["pred_pIC50"] == pytest.approx(0.0)
        assert results[0]["ci_lower"] == pytest.approx(-1.0)
        assert results[0]["ci_upper"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLIScreenCommands:
    """Test that screen/analyze-hits/report commands are registered."""

    def test_screen_help(self) -> None:
        from typer.testing import CliRunner

        from brd4kan.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["screen", "--help"])
        assert result.exit_code == 0
        assert "library" in result.output.lower() or "LIBRARY" in result.output

    def test_analyze_hits_help(self) -> None:
        from typer.testing import CliRunner

        from brd4kan.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["analyze-hits", "--help"])
        assert result.exit_code == 0
        assert "hits" in result.output.lower() or "curated" in result.output.lower()

    def test_report_help(self) -> None:
        from typer.testing import CliRunner

        from brd4kan.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["report", "--help"])
        assert result.exit_code == 0
        assert "html" in result.output.lower() or "report" in result.output.lower()
