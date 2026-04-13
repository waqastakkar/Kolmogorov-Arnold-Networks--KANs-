"""Stage 1 — Extract BRD4 binding activities from a local ChEMBL 36 SQLite DB.

The SQL applies every filter from PLAN.md §2 step 1:

* target ``CHEMBL1163125``
* binding assays only (``assay_type = 'B'``)
* ``confidence_score >= 8``
* ``standard_type ∈ {IC50, Ki, Kd}``
* ``standard_units = 'nM'``, ``standard_relation = '='``
* ``pchembl_value IS NOT NULL`` and ``standard_value IS NOT NULL``
* ``data_validity_comment IS NULL`` (sanity gate, additive to PLAN)

The output parquet plus a per-run manifest are written under
``artifacts/data/raw/`` (canonical) and
``artifacts/data/raw/runs/<sha>_<utc>/`` (manifest).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

from brd4kan.utils.config import Params
from brd4kan.utils.hashing import file_signature
from brd4kan.utils.manifest import (
    Manifest,
    env_snapshot,
    get_git_sha,
    utc_timestamp,
    write_manifest,
)
from brd4kan.utils.runs import make_run_dir

EXTRACT_SQL_TEMPLATE = """
SELECT
    md.chembl_id          AS molecule_chembl_id,
    cs.canonical_smiles   AS canonical_smiles,
    act.standard_type     AS standard_type,
    act.standard_relation AS standard_relation,
    act.standard_value    AS standard_value,
    act.standard_units    AS standard_units,
    act.pchembl_value     AS pchembl_value,
    ass.assay_id          AS assay_id,
    ass.confidence_score  AS confidence_score,
    ass.assay_type        AS assay_type,
    td.chembl_id          AS target_chembl_id,
    d.year                AS first_publication_year,
    d.doc_id              AS doc_id
FROM activities          act
JOIN assays              ass ON act.assay_id  = ass.assay_id
JOIN target_dictionary   td  ON ass.tid       = td.tid
JOIN compound_structures cs  ON act.molregno  = cs.molregno
JOIN molecule_dictionary md  ON act.molregno  = md.molregno
JOIN docs                d   ON act.doc_id    = d.doc_id
WHERE td.chembl_id = :target_chembl_id
  AND ass.assay_type = :assay_type
  AND ass.confidence_score >= :min_confidence_score
  AND act.standard_type IN ({type_placeholders})
  AND act.standard_units = :required_units
  AND act.standard_relation = :required_relation
  AND act.pchembl_value IS NOT NULL
  AND act.standard_value IS NOT NULL
  AND cs.canonical_smiles IS NOT NULL
  AND act.data_validity_comment IS NULL
"""


def build_extract_query(allowed_types: Sequence[str]) -> tuple[str, dict[str, str]]:
    """Inline an ``IN (?, ?, ...)`` placeholder list for the standard-type filter."""
    if not allowed_types:
        raise ValueError("allowed_standard_types must be non-empty")
    type_named = {f"type_{i}": t for i, t in enumerate(allowed_types)}
    placeholders_str = ", ".join(f":{k}" for k in type_named)
    sql = EXTRACT_SQL_TEMPLATE.format(type_placeholders=placeholders_str)
    return sql, type_named


def extract_activities(db_path: Path, params: Params) -> pd.DataFrame:
    """Run the extract SQL against ``db_path`` and return the result frame."""
    sql, type_named = build_extract_query(params.chembl.allowed_standard_types)
    bind: dict[str, str | int] = {
        "target_chembl_id": params.chembl.target_chembl_id,
        "assay_type": params.chembl.assay_type,
        "min_confidence_score": params.chembl.min_confidence_score,
        "required_units": params.chembl.required_units,
        "required_relation": params.chembl.required_relation,
        **type_named,
    }
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        df = pd.read_sql_query(sql, conn, params=bind)
    return df


def run_extract(out_dir: Path, params: Params, db_path: Path) -> Path:
    """End-to-end Stage 1: query → parquet → manifest."""
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = extract_activities(db_path, params)
    out_path = out_dir / "brd4_raw.parquet"
    df.to_parquet(out_path, index=False)

    run_dir = make_run_dir(out_dir)
    manifest = Manifest(
        stage="extract",
        git_sha=get_git_sha(),
        timestamp=utc_timestamp(),
        inputs={"chembl_db": file_signature(db_path)},
        outputs={
            "brd4_raw_parquet": file_signature(out_path),
            "n_rows": int(len(df)),
        },
        params_snapshot=params.model_dump(),
        seeds={"global": params.seed},
        env=env_snapshot(),
        wall_time_seconds=round(time.perf_counter() - started, 4),
    )
    write_manifest(manifest, run_dir)
    return out_path
