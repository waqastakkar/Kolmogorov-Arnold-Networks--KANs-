"""SQL filter correctness for Stage 1 (extract).

Builds a tiny ChEMBL-shaped SQLite (see ``conftest.tiny_chembl_db``) where
exactly one row is supposed to satisfy the canonical filter set, and asserts:

* the resulting frame contains exactly that one row,
* every filter clause is honored individually (parameterized counter-cases),
* the SQL builder rejects an empty allowed-types list,
* the placeholder list grows with the number of allowed types.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from brd4kan.data.extract import build_extract_query, extract_activities
from brd4kan.utils.config import Params


def test_build_extract_query_rejects_empty_types() -> None:
    with pytest.raises(ValueError):
        build_extract_query([])


def test_build_extract_query_emits_one_placeholder_per_type() -> None:
    sql, named = build_extract_query(["IC50", "Ki", "Kd"])
    assert set(named.values()) == {"IC50", "Ki", "Kd"}
    assert sql.count(":type_") == 3
    assert "act.standard_type IN" in sql


def test_extract_keeps_only_canonical_row(tiny_chembl_db: Path, params: Params) -> None:
    df = extract_activities(tiny_chembl_db, params)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["molecule_chembl_id"] == "CHEMBL_MOL_1"
    assert row["target_chembl_id"] == "CHEMBL1163125"
    assert row["standard_type"] == "IC50"
    assert row["standard_units"] == "nM"
    assert row["standard_relation"] == "="
    assert row["pchembl_value"] == pytest.approx(7.0)
    assert row["confidence_score"] >= params.chembl.min_confidence_score
    assert row["assay_type"] == params.chembl.assay_type
    assert row["first_publication_year"] == 2020


def test_every_counter_case_is_filtered_out(tiny_chembl_db: Path, params: Params) -> None:
    """The fixture has 8 deliberate fail rows (one per filter) plus 1 pass row.

    All 8 should be filtered out by the canonical SQL — only the pass row survives.
    """
    with sqlite3.connect(tiny_chembl_db) as conn:
        n_total_acts = int(pd.read_sql_query("SELECT COUNT(*) FROM activities", conn).iloc[0, 0])
    assert n_total_acts == 9, "fixture corruption: expected 9 source activities"
    df = extract_activities(tiny_chembl_db, params)
    assert len(df) == 1


def test_extract_returns_expected_columns(tiny_chembl_db: Path, params: Params) -> None:
    df = extract_activities(tiny_chembl_db, params)
    expected_cols = {
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "assay_id",
        "confidence_score",
        "assay_type",
        "target_chembl_id",
        "first_publication_year",
        "doc_id",
    }
    assert expected_cols.issubset(df.columns)
