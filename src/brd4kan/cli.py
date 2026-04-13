"""``brd4kan`` Typer CLI — one subcommand per pipeline stage (1-10 + report)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

from brd4kan.utils.config import Params, load_params
from brd4kan.utils.seed import set_global_seed

app = typer.Typer(
    add_completion=False,
    help="BRD4-KAN: reproducible symbolic QSAR & virtual screening pipeline.",
)


def _bootstrap(params_path: Optional[Path] = None) -> Params:
    load_dotenv()
    params = load_params(params_path)
    set_global_seed(params.seed)
    return params


def _resolve_db_path(params: Params, override: Optional[Path]) -> Path:
    if override is not None:
        return override
    raw = os.environ.get(params.chembl.db_env_var)
    if not raw:
        raise typer.BadParameter(
            f"ChEMBL DB path not set; export {params.chembl.db_env_var} in .env or pass --db"
        )
    return Path(raw)


@app.command()
def extract(
    out: Path = typer.Option(Path("artifacts/data/raw"), help="Output directory"),
    db: Optional[Path] = typer.Option(None, help="Path to chembl_36.db (overrides .env)"),
    params_path: Optional[Path] = typer.Option(None, "--params", help="Override params.yaml"),
) -> None:
    """Stage 1 — Extract BRD4 binding activities from local ChEMBL 36."""
    from brd4kan.data.extract import run_extract

    params = _bootstrap(params_path)
    db_path = _resolve_db_path(params, db)
    out_path = run_extract(out, params, db_path)
    typer.echo(str(out_path))


@app.command()
def curate(
    in_path: Path = typer.Option(
        Path("artifacts/data/raw/brd4_raw.parquet"), "--in", help="Raw parquet input"
    ),
    out: Path = typer.Option(Path("artifacts/data/processed"), help="Output directory"),
    params_path: Optional[Path] = typer.Option(None, "--params", help="Override params.yaml"),
) -> None:
    """Stage 2 — Standardize / filter / aggregate the raw extract."""
    from brd4kan.data.curate import run_curate

    params = _bootstrap(params_path)
    out_path = run_curate(in_path, out, params)
    typer.echo(str(out_path))


@app.command()
def split(
    in_path: Path = typer.Option(
        Path("artifacts/data/processed/brd4_curated.parquet"), "--in"
    ),
    out: Path = typer.Option(Path("artifacts/data/splits"), help="Output directory"),
    params_path: Optional[Path] = typer.Option(None, "--params"),
) -> None:
    """Stage 3 — Bemis-Murcko scaffold split + first-publication-year time split."""
    from brd4kan.data.split import run_split

    params = _bootstrap(params_path)
    paths = run_split(in_path, out, params)
    for k, p in paths.items():
        typer.echo(f"{k}: {p}")


@app.command()
def featurize(
    curated: Path = typer.Option(
        Path("artifacts/data/processed/brd4_curated.parquet"), help="Curated parquet"
    ),
    scaffold_split: Path = typer.Option(
        Path("artifacts/data/splits/scaffold_split.json"), help="Scaffold split JSON"
    ),
    out: Path = typer.Option(Path("artifacts/data/processed"), help="Output directory"),
    params_path: Optional[Path] = typer.Option(None, "--params"),
) -> None:
    """Stage 4 — Morgan + Mordred + Chemprop graph featurization."""
    from brd4kan.features.run import run_featurize

    params = _bootstrap(params_path)
    paths = run_featurize(curated, scaffold_split, out, params)
    for k, p in paths.items():
        typer.echo(f"{k}: {p}")


@app.command("train-baselines")
def train_baselines(
    curated: Path = typer.Option(
        Path("artifacts/data/processed/brd4_curated.parquet"), help="Curated parquet"
    ),
    scaffold_split: Path = typer.Option(
        Path("artifacts/data/splits/scaffold_split.json"), help="Scaffold split JSON"
    ),
    morgan: Path = typer.Option(
        Path("artifacts/data/processed/morgan.npz"), help="Morgan fingerprints"
    ),
    mordred: Path = typer.Option(
        Path("artifacts/data/processed/mordred.npz"), help="Mordred descriptors"
    ),
    out: Path = typer.Option(Path("artifacts/models/baselines"), help="Output directory"),
    params_path: Optional[Path] = typer.Option(None, "--params"),
) -> None:
    """Stage 5 — Optuna-tuned baselines (RF, XGBoost, SVR, MLP, Chemprop)."""
    from brd4kan.train.run_baselines import run_baselines

    params = _bootstrap(params_path)
    results = run_baselines(curated, scaffold_split, morgan, mordred, out, params)
    for model_name, info in results.items():
        if isinstance(info, dict) and "test_metrics" in info:
            rmse = info["test_metrics"].get("rmse_median", "?")
            typer.echo(f"{model_name}: RMSE_median={rmse}")
        else:
            typer.echo(f"{model_name}: {info}")


@app.command("train-kan")
def train_kan(
    curated: Path = typer.Option(
        Path("artifacts/data/processed/brd4_curated.parquet"), help="Curated parquet"
    ),
    scaffold_split: Path = typer.Option(
        Path("artifacts/data/splits/scaffold_split.json"), help="Scaffold split JSON"
    ),
    morgan: Path = typer.Option(
        Path("artifacts/data/processed/morgan.npz"), help="Morgan fingerprints"
    ),
    mordred: Path = typer.Option(
        Path("artifacts/data/processed/mordred.npz"), help="Mordred descriptors"
    ),
    out: Path = typer.Option(Path("artifacts/models/kan"), help="Output directory"),
    params_path: Optional[Path] = typer.Option(None, "--params"),
) -> None:
    """Stage 6 — Advanced KAN with ensemble, conformal, Optuna multi-objective."""
    from brd4kan.train.run_kan import run_kan

    params = _bootstrap(params_path)
    summary = run_kan(curated, scaffold_split, morgan, mordred, out, params)
    rmse_med = summary.get("aggregated", {}).get("rmse_median", "?")
    sparsity = summary.get("best_optuna_sparsity", "?")
    typer.echo(f"KAN: RMSE_median={rmse_med}, sparsity={sparsity}")


@app.command()
def symbolic(
    kan_model_dir: Path = typer.Option(
        Path("artifacts/models/kan"), help="KAN model directory with .pt checkpoints"
    ),
    curated: Path = typer.Option(
        Path("artifacts/data/processed/brd4_curated.parquet"), help="Curated parquet"
    ),
    morgan: Path = typer.Option(
        Path("artifacts/data/processed/morgan.npz"), help="Morgan fingerprints"
    ),
    mordred: Path = typer.Option(
        Path("artifacts/data/processed/mordred.npz"), help="Mordred descriptors"
    ),
    out: Path = typer.Option(Path("artifacts/reports/symbolic"), help="Output directory"),
    params_path: Optional[Path] = typer.Option(None, "--params"),
) -> None:
    """Stage 7 — Symbolic extraction: prune edges, fit closed-form equation."""
    from brd4kan.explain.symbolic import run_symbolic

    params = _bootstrap(params_path)
    result = run_symbolic(kan_model_dir, curated, morgan, mordred, out, params)
    typer.echo(f"Surviving edges: {result['n_surviving_edges']}")
    typer.echo(f"Top descriptor: {result['top_descriptors'][0]['name'] if result['top_descriptors'] else 'none'}")


@app.command()
def evaluate(
    curated: Path = typer.Option(
        Path("artifacts/data/processed/brd4_curated.parquet"), help="Curated parquet"
    ),
    scaffold_split: Path = typer.Option(
        Path("artifacts/data/splits/scaffold_split.json"), help="Scaffold split JSON"
    ),
    time_split: Path = typer.Option(
        Path("artifacts/data/splits/time_split.json"), help="Time split JSON"
    ),
    morgan: Path = typer.Option(
        Path("artifacts/data/processed/morgan.npz"), help="Morgan fingerprints"
    ),
    mordred: Path = typer.Option(
        Path("artifacts/data/processed/mordred.npz"), help="Mordred descriptors"
    ),
    baselines_dir: Path = typer.Option(
        Path("artifacts/models/baselines"), help="Baselines output dir"
    ),
    kan_dir: Path = typer.Option(
        Path("artifacts/models/kan"), help="KAN output dir"
    ),
    symbolic_dir: Path = typer.Option(
        Path("artifacts/reports/symbolic"), help="Symbolic extraction dir"
    ),
    figures_out: Path = typer.Option(
        Path("artifacts/figures/svg"), help="Figures output dir"
    ),
    metrics_out: Path = typer.Option(
        Path("artifacts/reports/metrics"), help="Metrics output dir"
    ),
    params_path: Optional[Path] = typer.Option(None, "--params"),
) -> None:
    """Stage 8 — Evaluation: metrics, bootstrap CIs, AD, SHAP, figures."""
    from brd4kan.train.run_evaluate import run_evaluate

    params = _bootstrap(params_path)
    results = run_evaluate(
        curated, scaffold_split, time_split, morgan, mordred,
        baselines_dir, kan_dir, symbolic_dir, figures_out, metrics_out, params,
    )
    for model_name, info in results.items():
        rmse = info.get("metrics", {}).get("rmse", "?")
        typer.echo(f"{model_name}: RMSE={rmse}")


@app.command()
def screen(
    library: Path = typer.Option(
        Path("artifacts/screening/library/library.smi"), help="SMILES file (one per line)"
    ),
    kan_dir: Path = typer.Option(
        Path("artifacts/models/kan"), help="KAN model directory"
    ),
    morgan: Path = typer.Option(
        Path("artifacts/data/processed/morgan.npz"), help="Morgan fingerprints (train)"
    ),
    mordred_scaler: Path = typer.Option(
        Path("artifacts/data/processed/mordred_scaler.joblib"), help="Mordred scaler"
    ),
    morgan_train: Path = typer.Option(
        Path("artifacts/data/processed/morgan.npz"), help="Morgan train fingerprints"
    ),
    mordred_train: Path = typer.Option(
        Path("artifacts/data/processed/mordred.npz"), help="Mordred train descriptors"
    ),
    top: int = typer.Option(500, help="Maximum number of top hits to select"),
    out: Path = typer.Option(Path("artifacts/screening/predictions"), help="Output directory"),
    params_path: Optional[Path] = typer.Option(None, "--params"),
) -> None:
    """Stage 9 -- Virtual screening: filter, featurize, predict, rank, diversify."""
    from brd4kan.screen.screening import run_screen

    params = _bootstrap(params_path)
    if top != 500:
        params.screen.default_top_n = top
    result = run_screen(library, kan_dir, morgan, mordred_scaler, morgan_train, mordred_train, out, params)
    typer.echo(f"Hits: {result.get('n_hits', 0)}")


@app.command("analyze-hits")
def analyze_hits(
    top_hits_csv: Path = typer.Option(
        Path("artifacts/screening/predictions/top_hits.csv"), help="Top-hits CSV from Stage 9"
    ),
    curated: Path = typer.Option(
        Path("artifacts/data/processed/brd4_curated.parquet"), help="Curated parquet"
    ),
    out: Path = typer.Option(Path("artifacts/screening/top_hits"), help="Output directory"),
    params_path: Optional[Path] = typer.Option(None, "--params"),
) -> None:
    """Stage 10 -- Annotate hits: nearest neighbor, novelty, pharmacophore, report cards."""
    from brd4kan.screen.analyze_hits import run_analyze_hits

    params = _bootstrap(params_path)
    result = run_analyze_hits(top_hits_csv, curated, out, params)
    typer.echo(f"Annotated {result['n_hits']} hits ({result['n_novel']} novel, "
               f"{result['n_pharmacophore']} pharmacophore match)")


@app.command()
def report(
    metrics_dir: Path = typer.Option(
        Path("artifacts/reports/metrics"), help="Metrics directory"
    ),
    figures_dir: Path = typer.Option(
        Path("artifacts/figures/svg"), help="Figures directory"
    ),
    symbolic_dir: Path = typer.Option(
        Path("artifacts/reports/symbolic"), help="Symbolic directory"
    ),
    hits_dir: Path = typer.Option(
        Path("artifacts/screening/top_hits"), help="Annotated hits directory"
    ),
    out: Path = typer.Option(
        Path("artifacts/reports/report.html"), help="Output HTML path"
    ),
    params_path: Optional[Path] = typer.Option(None, "--params"),
) -> None:
    """Generate self-contained HTML report aggregating all pipeline outputs."""
    from brd4kan.screen.report import build_report

    _bootstrap(params_path)
    path = build_report(metrics_dir, figures_dir, symbolic_dir, hits_dir, Path(out))
    typer.echo(f"Report: {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
