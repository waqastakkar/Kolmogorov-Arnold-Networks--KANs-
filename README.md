# BRD4-KAN: Interpretable Symbolic QSAR and Virtual Screening for BRD4 Inhibitors Using Kolmogorov-Arnold Networks

[![CI](https://github.com/YOUR_USERNAME/brd4-kan/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/brd4-kan/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Abstract

**BRD4-KAN** is a fully reproducible, end-to-end quantitative structure-activity relationship (QSAR) and virtual screening pipeline for bromodomain-containing protein 4 (BRD4, ChEMBL target [CHEMBL1163125](https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL1163125/)), built on advanced Kolmogorov-Arnold Networks (KANs). The pipeline extracts and curates BRD4 binding data from ChEMBL 36, trains an ensemble of spline-based KAN regressors with learnable activation functions, extracts a closed-form symbolic pIC50 equation via automated symbolic regression, and performs prospective virtual screening with calibrated uncertainty quantification.

Key methodological contributions:

- **Kolmogorov-Arnold Networks** with grid-extension schedule [3 &rarr; 5 &rarr; 10 &rarr; 20], multiplicative interaction nodes, and entropy-regularised sparsification for inherently interpretable QSAR.
- **Deep ensemble (n = 5) + MC-Dropout** uncertainty quantification with **Mondrian conformal prediction** stratified by Bemis-Murcko scaffold for calibrated per-compound prediction intervals.
- **Automated symbolic extraction** that distils the trained KAN into a human-readable, closed-form pIC50 = f(descriptors) equation.
- **Rigorous benchmarking** against Random Forest, XGBoost, SVR, MLP, and Chemprop D-MPNN baselines, all Optuna-tuned with scaffold-stratified cross-validation.
- **Full provenance tracking**: every run records parameter snapshots, input/output hashes, environment fingerprints, git SHAs, and wall-clock times in structured manifests.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Installation](#2-installation)
3. [Repository Structure](#3-repository-structure)
4. [Configuration](#4-configuration)
5. [Pipeline Overview](#5-pipeline-overview)
6. [Step-by-Step Usage](#6-step-by-step-usage)
   - [Stage 1: Extract](#stage-1--extract)
   - [Stage 2: Curate](#stage-2--curate)
   - [Stage 3: Split](#stage-3--split)
   - [Stage 4: Featurize](#stage-4--featurize)
   - [Stage 5: Train Baselines](#stage-5--train-baselines)
   - [Stage 6: Train KAN](#stage-6--train-kan)
   - [Stage 7: Symbolic Extraction](#stage-7--symbolic-extraction)
   - [Stage 8: Evaluation](#stage-8--evaluation)
   - [Stage 9: Virtual Screening](#stage-9--virtual-screening)
   - [Stage 10: Hit Analysis](#stage-10--hit-analysis)
   - [Report Generation](#report-generation)
7. [Python API](#7-python-api)
8. [Reproducibility](#8-reproducibility)
9. [Testing](#9-testing)
10. [Figures](#10-figures)
11. [Data Availability](#11-data-availability)
12. [Citation](#12-citation)
13. [License](#13-license)

---

## 1. System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| GPU | Not required | NVIDIA GPU with CUDA 12.1 (for KAN training acceleration) |
| Disk | 10 GB free | 50 GB free (for ChEMBL database + artifacts) |

### Software

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.11.x | Runtime |
| PyTorch | 2.4.1 | KAN backbone, ensemble training |
| CUDA | 12.1 (optional) | GPU acceleration |
| RDKit | 2024.03 | Cheminformatics (standardisation, fingerprints, 3D embedding) |
| ChEMBL Structure Pipeline | 1.2.2 | SMILES standardisation, desalting, neutralisation |
| mordredcommunity | 2.0.6 | 2D molecular descriptors |
| efficient-kan | 0.1.0 | KAN layer implementation |
| pykan | 0.2.8 | Symbolic regression stage |

A complete dependency list with pinned versions is in [`pyproject.toml`](pyproject.toml). Conda users: see [`environment.yml`](environment.yml).

### Tested Platforms

- Ubuntu 22.04 LTS (x86_64), Python 3.11.9, CUDA 12.1
- Windows 11 (x86_64), Python 3.11.9, CPU-only
- macOS 14 Sonoma (arm64), Python 3.11.9, CPU-only

Typical install time: **5-10 minutes** (pip) or **10-15 minutes** (conda).

---

## 2. Installation

### Option A: uv (recommended)

```bash
git clone https://github.com/YOUR_USERNAME/brd4-kan.git
cd brd4-kan

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install
uv venv --python 3.11
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

uv pip install -e ".[dev]"
uv run pre-commit install
```

### Option B: Conda

```bash
git clone https://github.com/YOUR_USERNAME/brd4-kan.git
cd brd4-kan
conda env create -f environment.yml
conda activate brd4-kan
pip install -e ".[dev]"
pre-commit install
```

### Option C: Make

```bash
make setup   # runs uv venv + install + pre-commit
```

### ChEMBL Database Setup

Download the [ChEMBL 36 SQLite database](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/) (~4.5 GB) and set the path:

```bash
echo 'CHEMBL_DB_PATH=/path/to/chembl_36.db' > .env
```

---

## 3. Repository Structure

```
brd4-kan/
├── PLAN.md                         # Full project specification
├── README.md                       # This file
├── pyproject.toml                  # Dependencies, build config, tool settings
├── environment.yml                 # Conda mirror
├── Makefile                        # Developer interface
├── dvc.yaml                        # DVC pipeline (10 stages)
├── params.yaml                     # Single source of truth: all hyperparameters
│
├── configs/                        # Hydra configuration overrides
│   ├── data.yaml
│   ├── model.yaml
│   ├── train.yaml
│   ├── screen.yaml
│   └── figures.yaml                # Nature MI figure style specification
│
├── src/brd4kan/                    # Main package
│   ├── __init__.py                 # Exports BRD4Predictor
│   ├── cli.py                      # Typer CLI (11 commands)
│   ├── predict.py                  # BRD4Predictor high-level API
│   │
│   ├── data/                       # Stages 1-3
│   │   ├── extract.py              # ChEMBL SQL extraction
│   │   ├── curate.py               # Standardisation, filtering, aggregation
│   │   └── split.py                # Scaffold + time splits
│   │
│   ├── features/                   # Stage 4
│   │   ├── morgan.py               # Morgan ECFP4 fingerprints
│   │   ├── mordred.py              # Mordred 2D descriptors + scaler
│   │   ├── graphs.py               # Chemprop-compatible CSV
│   │   └── run.py                  # Featurisation orchestrator
│   │
│   ├── models/                     # Model architectures
│   │   ├── baselines.py            # RF, XGBoost, SVR, MLP factories
│   │   ├── chemprop_wrapper.py     # Chemprop D-MPNN wrapper
│   │   ├── kan_model.py            # BRD4KANModel + EnsembleKAN
│   │   └── conformal.py            # Mondrian conformal predictor
│   │
│   ├── train/                      # Training infrastructure
│   │   ├── train_kan.py            # Single-member KAN training loop
│   │   ├── run_kan.py              # Stage 6 orchestrator (Optuna + ensemble)
│   │   ├── run_baselines.py        # Stage 5 orchestrator
│   │   ├── run_evaluate.py         # Stage 8 orchestrator
│   │   ├── cv.py                   # Scaffold-stratified k-fold CV
│   │   ├── metrics.py              # RMSE, MAE, R2, Spearman, ROC-AUC, etc.
│   │   ├── bootstrap.py            # 1000x bootstrap confidence intervals
│   │   ├── applicability.py        # Tanimoto + PCA-KDE applicability domain
│   │   └── mlflow_utils.py         # MLflow logging helpers
│   │
│   ├── explain/                    # Interpretability
│   │   ├── symbolic.py             # Stage 7: symbolic equation extraction
│   │   └── shap_analysis.py        # TreeSHAP + KernelSHAP
│   │
│   ├── screen/                     # Stages 9-10
│   │   ├── screening.py            # Virtual screening pipeline
│   │   ├── analyze_hits.py         # Hit annotation + pharmacophore
│   │   └── report.py               # HTML report generator
│   │
│   ├── viz/                        # Figures
│   │   ├── style.py                # Nature MI style enforcement
│   │   └── figures.py              # 9 figure types
│   │
│   └── utils/                      # Infrastructure
│       ├── config.py               # Pydantic-validated params.yaml loader
│       ├── seed.py                 # Global seed setter (random, numpy, torch)
│       ├── hashing.py              # SHA-256 for files and arrays
│       ├── manifest.py             # Per-run provenance manifest
│       └── runs.py                 # Timestamped run directory creation
│
├── scripts/                        # DVC entrypoints
│   ├── 01_extract.py ... 10_analyze_hits.py
│
├── tests/                          # pytest suite (>=85% coverage)
│   ├── conftest.py                 # Shared fixtures
│   ├── test_smoke_e2e.py           # Full pipeline integration test
│   ├── test_pipeline.py            # Stage 1-4 integration
│   ├── test_extract.py ... test_screen.py  # Per-module unit tests
│   └── test_style.py              # Figure style compliance
│
└── artifacts/                      # DVC-tracked outputs (gitignored)
    ├── data/{raw,processed,splits}
    ├── models/{kan,baselines}
    ├── reports/{metrics,shap,symbolic}
    ├── figures/svg/
    └── screening/{library,predictions,top_hits}
```

---

## 4. Configuration

All hyperparameters, thresholds, and paths are defined in [`params.yaml`](params.yaml). **No magic numbers exist in the source code.** Key sections:

| Section | Parameters | Description |
|---------|-----------|-------------|
| `seed` | `42` | Global random seed for full reproducibility |
| `chembl` | Target ID, assay filters, units | ChEMBL extraction criteria |
| `curate` | MW range, PAINS, replicate &sigma; | Data curation rules |
| `split` | Train/val/test fractions | Scaffold and time split configuration |
| `featurize` | Morgan radius/bits, Mordred filters | Feature engineering settings |
| `baselines` | Optuna trials, CV folds, n_seeds | Baseline model tuning |
| `kan` | Grid schedule, spline order, ensemble size, &lambda; values | KAN architecture and regularisation |
| `conformal` | &alpha;, strategy, partition_by | Conformal prediction settings |
| `symbolic` | Importance threshold, candidate functions | Symbolic extraction configuration |
| `evaluate` | Bootstrap iterations, AD method | Evaluation and uncertainty settings |
| `screen` | Top-N, QED min, Ro5, PAINS, clustering cutoff | Virtual screening filters |

Override any parameter at runtime:
```bash
brd4kan train-kan --params custom_params.yaml
```

---

## 5. Pipeline Overview

The pipeline consists of 10 stages organised as a directed acyclic graph (DAG), managed by [DVC](https://dvc.org/):

```
Extract (1) --> Curate (2) --> Split (3) --> Featurize (4) --+--> Baselines (5) --+
                                                             |                    |
                                                             +--> KAN (6) --------+--> Evaluate (8)
                                                             |                    |
                                                             +--> Symbolic (7) ---+
                                                                                  |
                                                             Screen (9) --> Analyze Hits (10)
```

Run the full pipeline:
```bash
dvc repro          # Execute all stages in dependency order
# or
make repro
```

Run a single stage:
```bash
dvc repro evaluate    # Run only the evaluate stage (+ dependencies)
```

---

## 6. Step-by-Step Usage

### Stage 1 &mdash; Extract

**What it does.** Queries a local ChEMBL 36 SQLite database for BRD4 (CHEMBL1163125) binding assay activities. Applies strict inclusion criteria: binding assays only (type B), confidence score &ge; 8, standard types &isin; {IC50, Ki, Kd}, units = nM, relation = "=", pchembl_value not null, no data validity comments.

**How it works.** A parameterised SQL template joins the `activities`, `assays`, `target_dictionary`, `compound_structures`, `molecule_dictionary`, and `docs` tables. Named placeholders are generated for the `standard_type IN (...)` clause. The query is executed via a read-only SQLite URI connection to prevent accidental writes.

**Output.** `artifacts/data/raw/brd4_raw.parquet` &mdash; raw activity records containing SMILES, pchembl_value, publication year, and ChEMBL identifiers, plus a provenance manifest.

**Command.**
```bash
# Via CLI
brd4kan extract --db /path/to/chembl_36.db

# Via DVC (uses CHEMBL_DB_PATH from .env)
dvc repro extract

# With custom output directory
brd4kan extract --db /path/to/chembl_36.db --out artifacts/data/raw
```

**Expected runtime.** ~30 seconds on a standard SSD.

---

### Stage 2 &mdash; Curate

**What it does.** Applies the ChEMBL Structure Pipeline to standardise, desalt, neutralise, and convert to parent form. Rejects molecules outside the drug-like property space: MW &notin; [150, 700], heavy atoms < 10, inorganic compounds, and mixtures. Flags PAINS substructures (retained but tagged). Aggregates replicate measurements per InChIKey using the median pIC50, discarding compounds with replicate &sigma; > 0.5 log units.

**How it works.**
1. Each SMILES is passed through `chembl_structure_pipeline.standardize_mol()` and `get_parent_mol()`.
2. Property filters are applied using RDKit descriptors (`MolWt`, `NumHeavyAtoms`).
3. Tautomer canonicalisation via RDKit `TautomerCanonicalizer`.
4. PAINS filtering uses RDKit's `FilterCatalog` with the PAINS catalogue.
5. InChIKey-grouped aggregation computes median pIC50 and replicate standard deviation.
6. An `active` binary label is assigned at pIC50 &ge; 6.5 (configurable in `params.yaml`).

**Output.** `artifacts/data/processed/brd4_curated.parquet`

**Command.**
```bash
brd4kan curate

# With explicit input
brd4kan curate --in artifacts/data/raw/brd4_raw.parquet --out artifacts/data/processed
```

**Expected runtime.** ~2 minutes for ~5,000 compounds.

---

### Stage 3 &mdash; Split

**What it does.** Generates two independent train/validation/test splits:
1. **Scaffold split** (80/10/10): Groups compounds by Bemis-Murcko generic scaffold. Scaffolds are sorted by group size (largest first) and assigned deterministically to train, then validation, then test until target fractions are met. Acyclic molecules receive unique singleton keys (`__singleton_{idx}__`) to prevent false leakage.
2. **Time split**: Compounds published before the 90th-percentile year go to train; the rest to test. This provides an out-of-time generalization assessment.

**How it works.** The scaffold split uses `rdkit.Chem.Scaffolds.MurckoScaffold.MakeScaffoldGeneric()` for scaffold extraction. The split algorithm is fully deterministic given the same input ordering and seed. Zero scaffold leakage is guaranteed by construction and verified by automated tests.

**Output.** `artifacts/data/splits/scaffold_split.json`, `artifacts/data/splits/time_split.json`

Each JSON contains index arrays: `{"train": [...], "val": [...], "test": [...]}`.

**Command.**
```bash
brd4kan split

# Verify zero leakage
python -c "
import json
s = json.load(open('artifacts/data/splits/scaffold_split.json'))
assert set(s['train']).isdisjoint(set(s['test'])), 'Leakage!'
print(f'Train: {len(s[\"train\"])}, Val: {len(s[\"val\"])}, Test: {len(s[\"test\"])}')
print('Zero scaffold leakage confirmed.')
"
```

**Expected runtime.** < 10 seconds.

---

### Stage 4 &mdash; Featurize

**What it does.** Computes three feature representations, each cached as compressed arrays:

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| **Morgan ECFP4** | n &times; 2048 | Circular fingerprints (radius 2, 2048 bits) |
| **Mordred 2D** | n &times; d | 2D molecular descriptors after variance filter (&sigma; = 0), correlation filter (\|&rho;\| > 0.95), and z-score normalisation |
| **Chemprop CSV** | n &times; 2 | SMILES + pIC50 for D-MPNN baseline |

**How it works.** Morgan fingerprints are computed with `rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect()`. Mordred descriptors are computed via the `mordredcommunity` calculator with `ignore_3D=True`. The Mordred scaler is **fitted on the training set only** (using scaffold split indices) and applied to the full dataset to prevent data leakage. Descriptors with zero variance or pairwise correlation |&rho;| > 0.95 are removed.

**Output.**
- `artifacts/data/processed/morgan.npz` (uint8 bit vectors)
- `artifacts/data/processed/mordred.npz` (float32 descriptors + column names)
- `artifacts/data/processed/mordred_scaler.joblib` (fitted StandardScaler)
- `artifacts/data/processed/chemprop.csv`

**Command.**
```bash
brd4kan featurize
```

**Expected runtime.** ~5 minutes for ~5,000 compounds.

---

### Stage 5 &mdash; Train Baselines

**What it does.** Trains and Optuna-tunes five baseline regressors for benchmarking against the KAN:

| Model | Type | Feature Input | Hyperparameter Search |
|-------|------|---------------|----------------------|
| Random Forest | Ensemble, bagging | Morgan + Mordred | n_estimators, max_depth, min_samples_split |
| XGBoost | Gradient boosting | Morgan + Mordred | n_estimators, max_depth, learning_rate, subsample |
| SVR | Kernel method | Morgan + Mordred (scaled) | C, epsilon, gamma, kernel |
| MLP | Neural network | Morgan + Mordred (scaled) | hidden_layer_sizes, alpha, learning_rate_init |
| Chemprop D-MPNN | Message-passing GNN | Raw SMILES | depth, hidden_size, ffn_num_layers, dropout |

**How it works.**
1. For each model, an Optuna study (100 trials) with Hyperband pruning searches the hyperparameter space using 5-fold scaffold-stratified cross-validation within the training set.
2. The best configuration is retrained on the full training set with 5 different random seeds.
3. Each seed's model is evaluated on the scaffold-test set.
4. Metrics are aggregated across seeds (median &plusmn; std).
5. All models, Optuna studies (.db), and metrics are logged to MLflow.

**Output.**
- `artifacts/models/baselines/<model>/seed_*.joblib` (trained models)
- `artifacts/models/baselines/<model>/best_hparams.json`
- `artifacts/models/baselines/<model>/test_metrics.json`
- `artifacts/models/baselines/baselines_summary.json`
- `artifacts/models/baselines/<model>_optuna.db`

**Command.**
```bash
brd4kan train-baselines

# View MLflow results
mlflow ui --backend-store-uri file:artifacts/mlflow
```

**Expected runtime.** ~30 minutes (CPU), depending on dataset size.

---

### Stage 6 &mdash; Train KAN

**What it does.** Trains an advanced Kolmogorov-Arnold Network ensemble with the following architecture:

```
[Input: Morgan + Mordred]
    --> MultiplicativeLayer (gated element-wise interactions: sigmoid(Wx_1) * Wx_2)
    --> KANLinear (learnable B-spline activations, grid schedule [3->5->10->20])
    --> ... (configurable depth: [64,1], [128,1], [128,64,1], [256,64,1])
    --> Regression head (pIC50)
    --> Auxiliary classification head (active >= 6.5, BCE loss)
```

**How it works.**
1. **Optuna multi-objective optimisation** (200 trials): simultaneously minimises RMSE and maximises network sparsity (measured by mean absolute spline weight), producing a Pareto front. Search space includes layer widths, grid size, learning rate, weight decay, dropout, batch size, &lambda; (L1), &lambda;<sub>ent</sub> (entropy), and optimiser choice (AdamW vs L-BFGS).
2. The best Pareto-front configuration (by RMSE) is retrained with **5 seeds &times; 5 ensemble members = 25 models**.
3. Training uses MSE loss + BCE (auxiliary) + &lambda;<sub>1</sub>&middot;L1 + &lambda;<sub>ent</sub>&middot;entropy regularisation, cosine learning rate schedule, gradient clipping (max norm 1.0), and early stopping on validation RMSE (patience 20 epochs).
4. **Grid extension** refines spline resolutions mid-training at epoch fractions: [3 &rarr; 5 &rarr; 10 &rarr; 20].
5. **Mondrian conformal prediction** is calibrated on the validation set, stratified by Bemis-Murcko scaffold, to provide per-compound prediction intervals at the (1 - &alpha;) = 90% confidence level. Unseen scaffold groups fall back to the global quantile.

**Output.**
- `artifacts/models/kan/best_hparams.json`
- `artifacts/models/kan/seed_*/member_*.pt` (ensemble checkpoints)
- `artifacts/models/kan/seed_*/conformal.json` (conformal calibration state)
- `artifacts/models/kan/kan_summary.json`
- `artifacts/models/kan/kan_optuna.db`

**Command.**
```bash
brd4kan train-kan

# On GPU
BRD4KAN_DEVICE=cuda brd4kan train-kan
```

**Expected runtime.** ~2-4 hours (GPU) or ~8-12 hours (CPU).

---

### Stage 7 &mdash; Symbolic Extraction

**What it does.** Distils the trained KAN into a human-readable closed-form equation: pIC50 = f(descriptors). This is the key interpretability contribution &mdash; unlike black-box deep learning, the KAN's learned spline activations can be approximated by elementary mathematical functions.

**How it works.**
1. **Edge importance estimation**: Computes mean absolute spline weight magnitude (`scaled_spline_weight`) for each input-to-first-layer edge across a 500-compound random subsample. Falls back to gradient-based importance if spline weights are unavailable.
2. **Pruning**: Removes edges with importance below the configured threshold (default: 0.01).
3. **Symbolic fitting**: For each surviving edge, fits six candidate functions using `scipy.optimize.curve_fit`:

   | Function | Form | Parameters |
   |----------|------|-----------|
   | poly2 | ax&sup2; + bx + c | 3 |
   | poly3 | ax&sup3; + bx&sup2; + cx + d | 4 |
   | exp | a&middot;exp(bx) | 2 |
   | log | a&middot;ln(\|x\| + &epsilon;) + b | 2 |
   | sin | a&middot;sin(bx + c) | 3 |
   | sigmoid | a / (1 + exp(-b(x - c))) | 3 |
   | sqrt | a&middot;&radic;(\|x\| + &epsilon;) + b | 2 |

   The best fit (by RMSE) is selected for each edge.

4. **Equation assembly**: Constructs a SymPy expression from the per-edge symbolic fits, with molecular descriptor names as symbols. Outputs LaTeX and a pickled SymPy expression.

**Output.**
- `artifacts/reports/symbolic/pIC50_equation.tex` (LaTeX equation)
- `artifacts/reports/symbolic/pIC50_equation.pkl` (SymPy expression, pickle)
- `artifacts/reports/symbolic/descriptor_importance.json` (ranked importance, top 50)
- `artifacts/reports/symbolic/symbolic_fits.json` (per-edge fit details)

**Command.**
```bash
brd4kan symbolic

# View the extracted equation
cat artifacts/reports/symbolic/pIC50_equation.tex
```

**Expected runtime.** ~1-5 minutes.

---

### Stage 8 &mdash; Evaluation

**What it does.** Comprehensive model evaluation with statistical rigour:

| Metric | Description | Type |
|--------|-------------|------|
| RMSE | Root mean squared error | Regression |
| MAE | Mean absolute error | Regression |
| R&sup2; | Coefficient of determination | Regression |
| Spearman &rho; | Rank correlation coefficient | Regression |
| Pearson r | Linear correlation coefficient | Regression |
| ROC-AUC | Receiver operating characteristic AUC | Classification (pIC50 &ge; 6.5) |
| PR-AUC | Precision-recall area under curve | Classification |
| MCC | Matthews correlation coefficient | Classification |
| Brier score | Probabilistic calibration | Classification |
| ECE | Expected calibration error | Classification |

All metrics are computed on both the scaffold-test and time-test splits, across 5 seeds, with **1000&times; bootstrap resampling** for 95% confidence intervals.

**How it works.**
1. Loads all trained baseline and KAN models; generates predictions on test sets.
2. Computes the full 10-metric suite per model.
3. Bootstrap resampling (1000 iterations, stratified) produces {mean, lo, hi, std} for each metric.
4. **Applicability domain** scoring via combined Tanimoto nearest-neighbour similarity (Morgan ECFP4, radius 2, 2048 bits) + PCA-KDE density estimation on the Mordred descriptor space.
5. Generates all publication-quality figures (see [Figures](#10-figures)).

**Output.**
- `artifacts/reports/metrics/evaluation_metrics.json`
- `artifacts/reports/metrics/ad_scores.json`
- `artifacts/figures/svg/01_dataset_overview.svg` ... `07_ad_map.svg`

**Command.**
```bash
brd4kan evaluate
```

**Expected runtime.** ~10-30 minutes (dominated by bootstrap resampling).

---

### Stage 9 &mdash; Virtual Screening

**What it does.** Screens a user-supplied compound library through the trained KAN ensemble with multi-stage filtering, uncertainty quantification, applicability domain assessment, and diversity selection.

**How it works.**
1. **Standardise** input SMILES via ChEMBL Structure Pipeline (standardise &rarr; desalt &rarr; neutralise &rarr; parent).
2. **Drug-likeness filter**: Lipinski Rule of Five (&le; 1 violation), QED &ge; 0.5, PAINS exclusion via RDKit FilterCatalog.
3. **Featurize**: Morgan ECFP4 (2048-bit) + Mordred 2D (using the fitted scaler from Stage 4).
4. **KAN ensemble inference**: Mean prediction &plusmn; epistemic (inter-model variance) and aleatoric (MC-Dropout, 50 samples) standard deviations.
5. **Conformal intervals**: Mondrian conformal prediction at the configured &alpha; level, stratified by predicted scaffold.
6. **AD scoring**: Tanimoto nearest-neighbour to training set + PCA-KDE density.
7. **Ranking**: Predicted pIC50 conditional on in-domain status (in-domain compounds receive a +100 rank boost).
8. **Diversity selection**: Butina clustering on Tanimoto distance matrix (cutoff 0.6), one representative per cluster, top-N selection (default: 500).
9. **3D embedding**: ETKDGv3 conformer generation + MMFF94s force-field minimisation (500 iterations) &rarr; dock-ready SDF with hydrogens removed.

**Output.**
- `artifacts/screening/predictions/screen_predictions.parquet` (full predictions)
- `artifacts/screening/predictions/top_hits.csv` (ranked top hits)
- `artifacts/screening/predictions/top_hits.sdf` (3D structures, dock-ready)

**Command.**
```bash
# Screen a SMILES file (one SMILES per line)
brd4kan screen --library my_library.smi --top 500

# Screen with custom top-N
brd4kan screen --library enamine_diversity.smi --top 1000
```

**Expected runtime.** ~10-60 minutes depending on library size.

---

### Stage 10 &mdash; Hit Analysis

**What it does.** Annotates each top hit with medicinal chemistry context for expert review:

| Annotation | Description |
|------------|-------------|
| Nearest ChEMBL neighbour | Most similar training compound (Tanimoto, Morgan ECFP4) + its experimental pIC50 |
| Novelty flag | `True` if Tanimoto to nearest training neighbour < 0.4 (structurally novel) |
| Scaffold class | Bemis-Murcko generic scaffold SMILES |
| Pharmacophore match | Presence of BRD4-relevant acetyl-lysine mimetic motifs |
| QED | Quantitative estimate of drug-likeness |
| SA score | Synthetic accessibility (1 = easy, 10 = hard) |

**How it works.** Nearest-neighbour search computes bulk Tanimoto similarity of Morgan fingerprints (radius 2, 2048 bits) against the full training set. Pharmacophore matching tests four SMARTS patterns representing acetyl-lysine mimetic pharmacophores critical for BRD4 bromodomain binding:

| Motif | SMARTS | Examples |
|-------|--------|----------|
| Amide-like | `[#7]~[#6](=[#8])~[#6]` | Acetamides, ureas |
| Indole/indazole | `c1[nH]c2ccccc2c1` | I-BET762 scaffold |
| Benzimidazole | `c1cc2[nH]ccc2cc1` | Molibresib-like |
| Triazine/pyrimidine | `[#7]1~[#6]~[#7]~[#6]~[#6]~1` | CPI-0610 scaffold |

**Output.**
- `artifacts/screening/top_hits/annotated_hits.csv`
- `artifacts/screening/top_hits/hit_cards.svg` (top-20 per-compound bar-chart summary)

**Command.**
```bash
brd4kan analyze-hits
```

**Expected runtime.** ~2-10 minutes.

---

### Report Generation

**What it does.** Generates a self-contained HTML report aggregating all pipeline outputs: metrics tables with bootstrap CIs, evaluation figures (inlined SVGs), the symbolic pIC50 equation, and the top-20 annotated hits table.

**Command.**
```bash
brd4kan report

# Custom output path
brd4kan report --out results/final_report.html
```

**Output.** `artifacts/reports/report.html` (self-contained, no external dependencies).

---

## 7. Python API

For programmatic access without the CLI:

```python
from brd4kan import BRD4Predictor

# Load trained model from artifacts
predictor = BRD4Predictor.load("artifacts/models/kan")

# Predict on new SMILES
results = predictor.predict_smiles([
    "Cc1cc(C)c2c(c1)C(=O)N(Cc1ccc(Cl)cc1)C2=O",
    "O=C(Nc1ccc2[nH]ncc2c1)c1ccc(Cl)cc1",
])

for r in results:
    print(f"SMILES:      {r['smiles'][:50]}...")
    print(f"  pIC50:     {r['pred_pIC50']:.2f}")
    print(f"  95% CI:    [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]")
    print(f"  Epistemic: {r['epistemic_std']:.3f}")
    print(f"  Aleatoric: {r['aleatoric_std']:.3f}")
    print(f"  In-domain: {r['ad_in_domain']}")
    print(f"  Tanimoto:  {r['tanimoto_nn']:.3f}")
    print()
```

**Return schema** (per compound):

| Key | Type | Description |
|-----|------|-------------|
| `smiles` | str | Input SMILES string |
| `pred_pIC50` | float | Ensemble mean predicted pIC50 |
| `ci_lower` | float | Conformal interval lower bound |
| `ci_upper` | float | Conformal interval upper bound |
| `epistemic_std` | float | Inter-model (epistemic) uncertainty |
| `aleatoric_std` | float | MC-Dropout (aleatoric) uncertainty |
| `ad_in_domain` | bool | Applicability domain membership |
| `tanimoto_nn` | float | Tanimoto similarity to nearest training compound |

---

## 8. Reproducibility

This pipeline implements the following reproducibility guarantees:

1. **Deterministic seeding.** `seed = 42` is set globally for Python `random`, `numpy.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and `PYTHONHASHSEED` before every stage. CUDA deterministic mode and `cudnn.benchmark = False` are enforced.

2. **Parameter provenance.** Every run records a complete `params.yaml` snapshot, git SHA, UTC timestamp, and `pip freeze` / `conda list` environment fingerprint in a structured `manifest.json`.

3. **Data hashing.** Input and output files are SHA-256 hashed (files < 200 MB) or fingerprinted by size + mtime (larger files). Array hashes are recorded in manifests for bit-level verification.

4. **Scaffold-split zero leakage.** The splitting algorithm guarantees that no Bemis-Murcko scaffold appears in more than one partition. This invariant is enforced by automated tests and verified in the smoke test.

5. **DVC pipeline.** All 10 stages are wired as a DVC DAG with explicit dependency and output declarations. `dvc repro` re-executes only stages whose inputs have changed.

6. **Locked dependencies.** `pyproject.toml` pins every dependency to an exact version. `environment.yml` provides a conda mirror.

7. **Run directories.** Each stage execution creates a timestamped directory under `<output_dir>/runs/<git_sha>_<utc>/` containing `manifest.json` and `git_sha.txt`.

To verify reproducibility:
```bash
# Run full pipeline twice and compare manifests
dvc repro
python -c "
import json, pathlib
for m in sorted(pathlib.Path('artifacts').rglob('manifest.json')):
    d = json.loads(m.read_text())
    print(f\"{d['stage']:>15s}  seed={d['seeds']['global']}  wall={d['wall_time_seconds']:.1f}s\")
"
```

---

## 9. Testing

The test suite comprises **18 test files** with **100+ test functions** covering unit tests, integration tests, and a comprehensive end-to-end smoke test.

```bash
# Run full test suite with coverage
make test
# or
pytest

# Run the end-to-end smoke test (Stages 1-10 + report)
make smoke
# or
pytest tests/test_smoke_e2e.py -v -x

# Run fast tests only (exclude slow integration)
pytest -m "not slow"

# Run tests for a specific stage
pytest tests/test_split.py -v
pytest tests/test_kan.py -v
pytest tests/test_screen.py -v
```

### Test Categories

| Category | Files | What is tested |
|----------|-------|----------------|
| Data pipeline | `test_extract.py`, `test_curate.py`, `test_split.py`, `test_featurize.py`, `test_graphs.py` | SQL filter correctness, curation invariants, zero-leakage scaffold splits, fingerprint determinism, Chemprop CSV round-trip |
| Models | `test_baselines.py`, `test_kan.py`, `test_symbolic.py` | Forward/backward shapes, ensemble uncertainty bounds, conformal coverage on synthetic data, symbolic function recovery (quadratic, exponential) |
| Evaluation | `test_evaluate.py`, `test_metrics.py` | Bootstrap CI bounds, perfect/imperfect predictions, single-class edge cases, AD fit/score shapes |
| Screening | `test_screen.py` | Drug-likeness filters, Butina clustering top-N, SDF validity, pharmacophore SMARTS matching, nearest-neighbour Tanimoto |
| Figures | `test_style.py`, `test_evaluate.py` | SVG format, Times New Roman font, bold weight, Nature NPG palette, despined axes |
| Integration | `test_pipeline.py`, `test_smoke_e2e.py` | End-to-end pipeline from synthetic data through report.html, manifest chain verification, array hash verification |
| Infrastructure | `test_utils.py`, `test_cli.py` | Config loading, seed determinism, SHA-256 hashing, manifest round-trip, all 11 CLI commands respond to --help |

Coverage gate: **&ge; 85%** (enforced via `pytest-cov --cov-fail-under=85`).

---

## 10. Figures

All figures comply with *Nature Machine Intelligence* style guidelines, enforced by `src/brd4kan/viz/style.py` and verified by `tests/test_style.py`:

| Rule | Implementation |
|------|---------------|
| Format | SVG only (`savefig(..., format='svg')`) |
| Font | Times New Roman, embedded (`svg.fonttype = 'none'`) |
| Weight | All text bold (`font.weight`, `axes.labelweight`, `axes.titleweight` = `'bold'`) |
| Sizes | Title 9 pt, axis label 8 pt, tick 7 pt, legend 7 pt, annotation 6 pt |
| Widths | 1-column 89 mm, 1.5-column 120 mm, 2-column 183 mm |
| Line width | 1.0 pt, axis spine 0.8 pt, tick length 3 pt |
| Palette | NPG: `#E64B35`, `#4DBBD5`, `#00A087`, `#3C5488`, `#F39B7F`, `#8491B4`, `#91D1C2`, `#DC0000`, `#7E6148`, `#B09C85` |
| Colormaps | Diverging `RdBu_r` for SHAP, sequential `viridis` for ordinal data |
| Axes | Despined (top + right spines removed) |
| Panels | Labelled **a**, **b**, **c**, ... bold 10 pt, top-left corner |

### Generated Figures

| # | Figure | Panels | Size |
|---|--------|--------|------|
| 1 | Dataset overview | (a) pIC50 histogram, (b) active/inactive bar, (c) scaffold UMAP | 2-col |
| 2 | Benchmark bars | (a) scaffold split, (b) time split, error bars = 95% bootstrap CI | 2-col |
| 3 | Parity + residual | (a) observed vs predicted scatter with identity line, (b) residual vs predicted | 2-col &times; model |
| 4 | KAN splines | (a) top-10 descriptor importance horizontal bars | 1.5-col |
| 5 | Symbolic equation | (a) rendered LaTeX equation, (b) descriptor importance ranking | 2-col |
| 6 | SHAP beeswarm | (a) feature-level SHAP values coloured by feature magnitude | 1.5-col &times; model |
| 7 | AD map | (a) PCA scatter, in-domain (viridis by pIC50) vs out-of-domain (x markers) | 1-col |
| 8 | Screening funnel | (a) compound attrition bars, (b) top-hit UMAP by predicted pIC50 | 2-col |
| 9 | Hit cards | (a-t) per-compound bars for pIC50, AD, QED, SA | 1-col |

---

## 11. Data Availability

- **ChEMBL 36**: Publicly available from the European Bioinformatics Institute at [https://www.ebi.ac.uk/chembl/](https://www.ebi.ac.uk/chembl/). The BRD4 target page is [CHEMBL1163125](https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL1163125/). Database DOI: [10.6019/CHEMBL.database.36](https://doi.org/10.6019/CHEMBL.database.36).
- **Screening libraries**: Users supply their own SMILES files via `--library`. The pipeline is compatible with [Enamine REAL](https://enamine.net/compound-collections/real-compounds), [ZINC](https://zinc.docking.org/), and other standard formats (one SMILES per line).
- **Processed data and trained models**: All intermediate artifacts are fully reproduced from raw data via `dvc repro`. Pre-computed artifacts can be shared via DVC remote storage (S3, GCS, SSH, etc.).

---

## 12. Citation

If you use this software in your research, please cite:

```bibtex
@software{brd4kan2024,
  title     = {{BRD4-KAN}: Interpretable Symbolic {QSAR} and Virtual Screening
               for {BRD4} Inhibitors Using {Kolmogorov-Arnold} Networks},
  author    = {{BRD4-KAN Authors}},
  year      = {2024},
  url       = {https://github.com/YOUR_USERNAME/brd4-kan},
  version   = {0.1.0},
  note      = {Software}
}
```

### Related References

1. Liu, Z. *et al.* KAN: Kolmogorov-Arnold Networks. *arXiv* 2404.19756 (2024).
2. Filipiak, P. *et al.* Efficient-KAN: efficient implementation of Kolmogorov-Arnold Networks. *GitHub* (2024).
3. Yang, K. *et al.* Analyzing learned molecular representations for property prediction. *J. Chem. Inf. Model.* **59**, 3370-3388 (2019).
4. Bemis, G.W. & Murcko, M.A. The properties of known drugs. 1. Molecular frameworks. *J. Med. Chem.* **39**, 2887-2893 (1996).
5. Mendez, D. *et al.* ChEMBL: towards direct deposition of bioassay data. *Nucleic Acids Res.* **47**, D930-D940 (2019).
6. Vovk, V. *et al.* Algorithmic Learning in a Random World. *Springer* (2005). [Conformal prediction]
7. Bender, A. *et al.* Molecular similarity searching using atom environments. *J. Chem. Inf. Comput. Sci.* **44**, 170-178 (2004).

---

## 13. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Generated by the BRD4-KAN pipeline. For questions or issues, please open a GitHub issue.*
