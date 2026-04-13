"""Pydantic-validated loader for ``params.yaml``.

``params.yaml`` is the single source of truth for hyperparameters, paths,
thresholds, and seeds. Every script must obtain its values through this
module — never hard-code them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PARAMS_PATH = _REPO_ROOT / "params.yaml"


class _BM(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=False)


class PathsConfig(_BM):
    raw_data: str
    interim: str
    processed: str
    splits: str
    models: str
    kan_models: str
    baseline_models: str
    optuna: str
    mlflow: str
    reports: str
    metrics: str
    shap: str
    symbolic: str
    figures: str
    screening: str
    library: str
    predictions: str
    top_hits: str


class ChemblConfig(_BM):
    db_env_var: str
    target_chembl_id: str
    assay_type: str
    min_confidence_score: int
    allowed_standard_types: list[str]
    required_units: str
    required_relation: str
    require_pchembl: bool


class CurateConfig(_BM):
    mw_min: float
    mw_max: float
    min_heavy_atoms: int
    reject_inorganic: bool
    reject_mixtures: bool
    replicate_sigma_threshold: float
    pains_flag_keep: bool
    active_pchembl_threshold: float


class ScaffoldSplitConfig(_BM):
    train_frac: float
    val_frac: float
    test_frac: float
    include_chirality: bool


class TimeSplitConfig(_BM):
    year_field: str
    test_year_quantile: float


class SplitConfig(_BM):
    scaffold: ScaffoldSplitConfig
    time: TimeSplitConfig


class MorganConfig(_BM):
    radius: int
    n_bits: int
    use_features: bool
    use_chirality: bool


class MordredConfig(_BM):
    ignore_3d: bool
    variance_threshold: float
    correlation_threshold: float
    nan_policy: str


class GraphsConfig(_BM):
    atom_messages: bool


class FeaturizeConfig(_BM):
    morgan: MorganConfig
    mordred: MordredConfig
    graphs: GraphsConfig


class BaselinesConfig(_BM):
    optuna_trials: int
    cv_folds: int
    n_seeds: int
    models: list[str]


class KANConfig(_BM):
    backbone: str
    grid_schedule: list[int]
    spline_order: int
    learnable_base: bool
    learnable_spline_scale: bool
    multiplicative_nodes: bool
    dropout: float
    aux_classification_head: bool
    aux_active_threshold: float
    lamb: float
    lamb_entropy: float
    lamb_coef: float
    ensemble_size: int
    mc_dropout_samples: int
    optuna_trials: int
    optuna_directions: list[str]
    pruner: str
    cv_folds: int
    n_seeds: int
    early_stopping_patience: int
    grad_clip: float
    lr_schedule: str


class ConformalConfig(_BM):
    alpha: float
    strategy: str
    partition_by: str


class SymbolicConfig(_BM):
    edge_importance_threshold: float
    candidate_functions: list[str]


class EvaluateConfig(_BM):
    bootstrap_iters: int
    ad_method: str
    tanimoto_radius: int
    tanimoto_nbits: int


class ScreenConfig(_BM):
    default_top_n: int
    ro5_enforce: bool
    qed_min: float
    pains_filter: bool
    cluster_cutoff: float
    diversity_selection: str
    embed_algorithm: str
    ff: str
    ff_max_iters: int


class ReportConfig(_BM):
    output: str


class Params(_BM):
    seed: int
    paths: PathsConfig
    chembl: ChemblConfig
    curate: CurateConfig
    split: SplitConfig
    featurize: FeaturizeConfig
    baselines: BaselinesConfig
    kan: KANConfig
    conformal: ConformalConfig
    symbolic: SymbolicConfig
    evaluate: EvaluateConfig
    screen: ScreenConfig
    report: ReportConfig


def load_params(path: Path | None = None) -> Params:
    """Read ``params.yaml`` (default: repo root) and return a validated :class:`Params`."""
    cfg_path = path or _DEFAULT_PARAMS_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    return Params.model_validate(raw)


def repo_root() -> Path:
    return _REPO_ROOT
