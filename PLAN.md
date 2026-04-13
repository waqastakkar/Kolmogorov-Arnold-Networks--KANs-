BRD4-KAN: A Reproducible Symbolic QSAR & Virtual Screening Pipeline
Below is a complete, Claude-Code-ready project plan. Hand it to Claude Code as PLAN.md at repo root and have it implement step by step.
0. Repo layout
brd4-kan/
├── PLAN.md                      ← this file
├── README.md
├── pyproject.toml               ← uv/poetry, pinned versions
├── environment.yml              ← conda mirror
├── Makefile                     ← `make all` runs full pipeline
├── dvc.yaml                     ← DVC stages, one per step below
├── params.yaml                  ← every hyperparameter, no magic numbers in code
├── configs/
│   ├── data.yaml  model.yaml  train.yaml  screen.yaml  figures.yaml
├── src/brd4kan/
│   ├── data/  features/  models/  train/  explain/  screen/  viz/  utils/
├── scripts/                     ← thin CLI entrypoints (Typer)
├── tests/                       ← pytest, ≥85% coverage gate
├── notebooks/                   ← read-only, generated from scripts
└── artifacts/                   ← gitignored, tracked by DVC
    ├── data/{raw,interim,processed,splits}
    ├── models/{kan,baselines}/<run_id>/
    ├── reports/{metrics,shap,symbolic}/
    ├── figures/svg/
    └── screening/{library,predictions,top_hits}/
Reproducibility contract: every script reads params.yaml, writes outputs under artifacts/<stage>/<git_sha>_<timestamp>/, logs to MLflow (local file backend), and emits a manifest.json (inputs, outputs, hashes, env, seeds, wall time). Global seed = 42, set for random, numpy, torch, torch.cuda, PYTHONHASHSEED. CUDA deterministic mode on. Conda-lock + uv.lock committed.
1. Environment & infra

Python 3.11, CUDA 12.1, PyTorch 2.4
Core: rdkit==2024.03, chembl_structure_pipeline, mordredcommunity, datamol, scikit-learn, xgboost, chemprop, efficient-kan, pykan (for symbolic stage), optuna, mlflow, dvc[all], hydra-core, typer, pydantic, pytest, ruff, mypy, pre-commit, matplotlib, cairosvg, svgutils.
make setup → installs, runs pre-commit, fetches local ChEMBL 36 path from .env.

2. Pipeline stages (each = one DVC stage, one CLI command, one test file)
Stage 1 — Extract (scripts/01_extract.py)
SQL against local chembl_36.db for CHEMBL1163125: binding assays only, confidence_score ≥ 8, standard_type ∈ {IC50, Ki, Kd}, units = nM, relation = '=', pchembl_value NOT NULL. Output artifacts/data/raw/brd4_raw.parquet + manifest.
Stage 2 — Curate (02_curate.py)
ChEMBL Structure Pipeline → standardize, desalt, neutralize, parent. Reject MW ∉ [150, 700], heavy atoms < 10, inorganics, mixtures. Tautomer canonicalization. PAINS flag (keep, tag). Aggregate by InChIKey: median pIC50, drop if replicate σ > 0.5. Output processed/brd4_curated.parquet.
Stage 3 — Split (03_split.py)
Bemis–Murcko scaffold split 80/10/10, plus a second time split by first_publication_year for an out-of-time generalization test. Save indices as JSON.
Stage 4 — Featurize (04_featurize.py)
Three feature views, each cached as .npz:

Morgan ECFP4 2048-bit
Mordred 2D → variance + |ρ|>0.95 filter → z-score (scaler pickled)
Graphs for Chemprop baseline

Stage 5 — Baselines (05_baselines.py)
RF, XGBoost, SVR (descriptors), Chemprop D-MPNN (graphs), MLP matched to KAN param count. Each: Optuna 100 trials, 5-fold CV inside train, 5 seeds on test. Save models + metrics.
Stage 6 — KAN (advanced) (06_train_kan.py)
Use the most advanced KAN features available:

efficient-kan backbone for speed; pyKAN wrapper for the symbolic stage
Grid extension schedule: grid sizes [3, 5, 10, 20], refined mid-training
Spline order k = 3, learnable base + spline scales
Entropy + L1 sparsification (lamb, lamb_entropy, lamb_coef)
Multiplicative KAN nodes (MultKAN) to capture feature interactions
Auxiliary classification head (active ≥ 6.5) with uncertainty via MC-Dropout + Deep Ensembles (n=5)
Conformal prediction wrapper (Mondrian, per scaffold) for calibrated intervals
Optuna 200 trials, multi-objective: minimize RMSE + maximize sparsity. Pruner: Hyperband. Search space in configs/model.yaml: layer widths, grid, k, λ's, lr, weight decay, dropout, batch size, optimizer (LBFGS vs AdamW)
Train with early stopping on val RMSE, gradient clipping, cosine LR
Save: best config, weights (.pt), full Optuna study (.db), MLflow run

Stage 7 — Symbolic extraction (07_symbolic.py)
Prune low-importance edges → auto_symbolic() fits each surviving spline to {poly, exp, log, sin, sigmoid, sqrt}. Emit a closed-form pIC50 equation (LaTeX + SymPy pickle), and a per-descriptor importance ranking.
Stage 8 — Evaluation (08_evaluate.py)
Metrics on scaffold-test and time-test, 5 seeds: RMSE, MAE, R², Spearman ρ, Pearson r, ROC-AUC, PR-AUC, MCC, Brier, ECE. Applicability domain via Tanimoto-to-train + KDE on descriptor PCA. Bootstrap 1000× CIs. SHAP for KAN + baselines for cross-comparison.
Stage 9 — Virtual screening (09_screen.py)
Library: Enamine REAL Diversity (or any user-supplied SMILES file via --library). Steps: standardize → drug-likeness (Ro5 + QED ≥ 0.5 + PAINS-out) → featurize → KAN ensemble inference with conformal intervals → rank by predicted pIC50 conditional on AD-in-domain → diversity selection via Butina clustering (Tanimoto 0.6) → top-N (default 500) with one representative per cluster → dock-ready SDF (3D embed, ETKDGv3, MMFF94s minimize).
Stage 10 — Screening analysis & hit selection (10_analyze_hits.py)
Per top-hit: predicted pIC50 ± conformal CI, AD score, nearest ChEMBL neighbor (Tanimoto + pIC50), scaffold class, novelty flag, key descriptors driving prediction (KAN edge attribution), QED, SA score, BRD4 pharmacophore match (acetyl-lysine mimetic SMARTS). Output ranked CSV + per-compound report cards (SVG).
3. Figures spec (Nature MI level)
Single module src/brd4kan/viz/ with one style.py enforcing every rule.
Hard rules (enforced in style.py, unit-tested):

Format: SVG only (savefig(..., format='svg'))
Font: Times New Roman, embedded; set font.family='serif', font.serif=['Times New Roman'], svg.fonttype='none'
All text bold (font.weight='bold', axes.labelweight='bold', axes.titleweight='bold')
Controlled sizes (configurable in figures.yaml): title 9pt, axis label 8pt, tick 7pt, legend 7pt, annotation 6pt
Line width 1.0, axis spine 0.8, tick length 3
Sizes: 1-col 89mm, 1.5-col 120mm, 2-col 183mm; height per panel
Nature color palette (Nature/NPG via ggsci-equivalent, hardcoded hex):
["#E64B35","#4DBBD5","#00A087","#3C5488","#F39B7F","#8491B4","#91D1C2","#DC0000","#7E6148","#B09C85"]
Plus diverging RdBu_r for SHAP, sequential viridis only when ordinal.
No chartjunk; despined top/right; panel letters a, b, c… bold 10pt top-left.

Figures produced:

Dataset overview (counts, pIC50 hist, scaffold UMAP)
Benchmark bars: KAN vs baselines, scaffold + time split, error bars = 95% bootstrap CI
Parity + residual plots
KAN architecture schematic + learned spline shapes for top-10 descriptors
Symbolic equation panel + descriptor importance
SHAP beeswarm (KAN vs RF vs Chemprop)
Applicability domain map (PCA + KDE)
Screening funnel + top-hit chemical space UMAP colored by predicted pIC50
Top-20 hit cards (structure + bars for pIC50, AD, QED, SA)

4. Hyperparameter tuning policy
Optuna study per model, persisted to artifacts/optuna/<model>.db. Search spaces in configs/model.yaml. Multi-objective (RMSE ↓, sparsity ↑) for KAN; single-objective RMSE for baselines. 5-fold scaffold CV inside train. Pruner: Hyperband. n_trials: baselines 100, KAN 200. Best config re-trained on full train set with 5 seeds.
5. Tool packaging
Expose a Typer CLI brd4kan:
brd4kan extract | curate | split | featurize | train-baselines | train-kan |
        symbolic | evaluate | screen --library file.smi --top 500 |
        analyze-hits | report
brd4kan run-all   # executes the DVC DAG
Plus a thin Python API: from brd4kan import BRD4Predictor; p = BRD4Predictor.load("artifacts/models/kan/best"); p.predict_smiles([...]) returning pIC50, conformal interval, AD flag, top contributing descriptors.
6. Testing & CI
pytest units for: SQL filter correctness, curation invariants, scaffold-split leakage check (no shared scaffolds across splits), featurizer determinism, KAN forward/backward shape, conformal coverage on synthetic data, figure-style compliance (font, weight, format, palette). GitHub Actions: lint (ruff), type (mypy), tests, dvc repro --dry. Coverage gate ≥ 85%.
7. What gets saved (everything)
Under each run dir: params.yaml snapshot, manifest.json, env.txt (pip freeze + conda list), git_sha.txt, raw + processed data hashes, scalers (.joblib), feature matrices (.npz), splits (.json), all model weights, Optuna .db, MLflow run, metrics (.json + .csv), SHAP values (.npz), symbolic equation (.tex + .pkl), screening predictions (.parquet), top hits (.csv + .sdf), all figures (.svg), and a final auto-built report.html aggregating everything.
8. Order of implementation for Claude Code

Repo skeleton + pyproject.toml + Makefile + params.yaml + style.py + tests for style.
Stages 1→4 with tests, then DVC wire-up.
Stage 5 baselines + Optuna + MLflow.
Stage 6 advanced KAN (efficient-kan + ensembles + conformal) + Optuna.
Stage 7 symbolic via pyKAN.
Stage 8 evaluation + all figures.
Stage 9 screening + Stage 10 hit analysis + report.html.
End-to-end smoke test on a 500-row subsample, then full run.