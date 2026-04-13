"""Shared test fixtures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from brd4kan.utils.config import load_params


@pytest.fixture(scope="session")
def params():
    return load_params()


@pytest.fixture()
def tiny_chembl_db(tmp_path: Path) -> Path:
    """A minimal ChEMBL-shaped SQLite covering only the columns the extract SQL touches.

    Rows are crafted so the canonical filter set keeps exactly **one** row
    (``CHEMBL_MOL_1``); every other row violates exactly one filter.
    """
    db_path = tmp_path / "tiny_chembl.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.executescript(
        """
        CREATE TABLE target_dictionary (
            tid INTEGER PRIMARY KEY,
            chembl_id TEXT
        );
        CREATE TABLE assays (
            assay_id INTEGER PRIMARY KEY,
            tid INTEGER,
            confidence_score INTEGER,
            assay_type TEXT,
            doc_id INTEGER
        );
        CREATE TABLE activities (
            activity_id INTEGER PRIMARY KEY,
            assay_id INTEGER,
            molregno INTEGER,
            doc_id INTEGER,
            standard_type TEXT,
            standard_relation TEXT,
            standard_value REAL,
            standard_units TEXT,
            pchembl_value REAL,
            data_validity_comment TEXT
        );
        CREATE TABLE compound_structures (
            molregno INTEGER PRIMARY KEY,
            canonical_smiles TEXT
        );
        CREATE TABLE molecule_dictionary (
            molregno INTEGER PRIMARY KEY,
            chembl_id TEXT
        );
        CREATE TABLE docs (
            doc_id INTEGER PRIMARY KEY,
            year INTEGER
        );
        """
    )

    c.execute("INSERT INTO target_dictionary VALUES (1, 'CHEMBL1163125')")
    c.execute("INSERT INTO target_dictionary VALUES (2, 'CHEMBL_OTHER')")

    # assay 10: passes (target=BRD4, conf=9, binding)
    # assay 11: fails confidence
    # assay 12: fails assay type (functional)
    # assay 13: wrong target
    c.executemany(
        "INSERT INTO assays VALUES (?, ?, ?, ?, ?)",
        [
            (10, 1, 9, "B", 100),
            (11, 1, 7, "B", 100),
            (12, 1, 9, "F", 100),
            (13, 2, 9, "B", 100),
        ],
    )

    c.execute("INSERT INTO compound_structures VALUES (1, 'CCN(CC)Cc1ccc(cc1)C(=O)O')")
    c.execute("INSERT INTO molecule_dictionary VALUES (1, 'CHEMBL_MOL_1')")
    c.execute("INSERT INTO docs VALUES (100, 2020)")

    # Activity matrix:
    #   1000: PASS — IC50/=/nM/pchembl=7.0/clean
    #   1001: FAIL — standard_type LogP not in {IC50,Ki,Kd}
    #   1002: FAIL — units uM not nM
    #   1003: FAIL — relation '>' not '='
    #   1004: FAIL — pchembl_value NULL
    #   1005: FAIL — data_validity_comment present
    #   1006: FAIL — uses assay 11 (low confidence)
    #   1007: FAIL — uses assay 12 (functional)
    #   1008: FAIL — uses assay 13 (wrong target)
    rows = [
        (1000, 10, 1, 100, "IC50", "=", 100.0, "nM", 7.0, None),
        (1001, 10, 1, 100, "LogP", "=", 1.0, "nM", 7.0, None),
        (1002, 10, 1, 100, "IC50", "=", 100.0, "uM", 7.0, None),
        (1003, 10, 1, 100, "IC50", ">", 100.0, "nM", 7.0, None),
        (1004, 10, 1, 100, "IC50", "=", 100.0, "nM", None, None),
        (1005, 10, 1, 100, "IC50", "=", 100.0, "nM", 7.0, "Outside typical range"),
        (1006, 11, 1, 100, "IC50", "=", 100.0, "nM", 7.0, None),
        (1007, 12, 1, 100, "IC50", "=", 100.0, "nM", 7.0, None),
        (1008, 13, 1, 100, "IC50", "=", 100.0, "nM", 7.0, None),
    ]
    c.executemany("INSERT INTO activities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return db_path
