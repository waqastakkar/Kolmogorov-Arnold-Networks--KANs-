"""Curation invariants for Stage 2.

These tests need RDKit + ChEMBL Structure Pipeline. They skip cleanly when
those deps are missing so the rest of the suite still runs in minimal envs.
"""

from __future__ import annotations

import pandas as pd
import pytest

rdkit = pytest.importorskip("rdkit")
csp = pytest.importorskip("chembl_structure_pipeline")

from brd4kan.data.curate import (  # noqa: E402
    aggregate_replicates,
    curate,
    is_inorganic_mol,
    is_mixture,
    passes_property_filters,
    standardize_smiles,
)
from brd4kan.utils.config import Params  # noqa: E402


def test_is_mixture() -> None:
    assert is_mixture("CCO.[Na+]")
    assert not is_mixture("CCO")


def test_is_inorganic_mol() -> None:
    from rdkit import Chem

    water = Chem.MolFromSmiles("O")
    assert is_inorganic_mol(water)
    ethanol = Chem.MolFromSmiles("CCO")
    assert not is_inorganic_mol(ethanol)


def test_standardize_idempotent() -> None:
    smi = "CCN(CC)CCNC(=O)c1ccc(N)cc1"
    once = standardize_smiles(smi)
    assert once is not None
    twice = standardize_smiles(once)
    assert once == twice


def test_passes_property_filters_mw_bounds(params: Params) -> None:
    from rdkit import Chem

    too_small = Chem.MolFromSmiles("CCO")  # MW ~46
    ok, reason = passes_property_filters(too_small, params)
    assert ok is False and reason and reason.startswith("mw_oor")

    bromodomain_like = Chem.MolFromSmiles("Cc1cc(C)c2c(c1)C(=O)N(Cc1ccc(Cl)cc1)C2=O")
    ok, _ = passes_property_filters(bromodomain_like, params)
    assert ok


def test_aggregate_replicates_takes_median_and_drops_high_sigma(params: Params) -> None:
    df = pd.DataFrame(
        [
            {"canonical_smiles_std": "c1ccccc1C(=O)O", "pchembl_value": 7.0},
            {"canonical_smiles_std": "c1ccccc1C(=O)O", "pchembl_value": 7.2},
            {"canonical_smiles_std": "c1ccccc1C(=O)O", "pchembl_value": 7.1},
            {"canonical_smiles_std": "CC(=O)Oc1ccccc1C(=O)O", "pchembl_value": 5.0},
            {"canonical_smiles_std": "CC(=O)Oc1ccccc1C(=O)O", "pchembl_value": 8.0},
        ]
    )
    out = aggregate_replicates(df, sigma_threshold=params.curate.replicate_sigma_threshold)
    benzoic = out[out["canonical_smiles_std"] == "c1ccccc1C(=O)O"]
    assert len(benzoic) == 1
    assert float(benzoic["pchembl_value"].iloc[0]) == pytest.approx(7.1, abs=1e-6)
    aspirin = out[out["canonical_smiles_std"] == "CC(=O)Oc1ccccc1C(=O)O"]
    assert len(aspirin) == 0  # σ ~1.5 > 0.5 → dropped


def test_full_curate_pipeline_drops_oor_keeps_drug_like(params: Params) -> None:
    raw = pd.DataFrame(
        [
            {
                "canonical_smiles": "CCO",
                "pchembl_value": 6.0,
                "first_publication_year": 2018,
                "molecule_chembl_id": "CHEMBL_TINY",
            },
            {
                "canonical_smiles": "Cc1cc(C)c2c(c1)C(=O)N(Cc1ccc(Cl)cc1)C2=O",
                "pchembl_value": 7.5,
                "first_publication_year": 2020,
                "molecule_chembl_id": "CHEMBL_DRUG_1",
            },
            {
                "canonical_smiles": "[Na+].[Cl-]",
                "pchembl_value": 6.0,
                "first_publication_year": 2018,
                "molecule_chembl_id": "CHEMBL_SALT",
            },
        ]
    )
    curated = curate(raw, params)
    assert "CHEMBL_DRUG_1" in curated["molecule_chembl_id"].tolist()
    assert "CHEMBL_TINY" not in curated["molecule_chembl_id"].tolist()
    assert "active" in curated.columns
    assert "pains_flag" in curated.columns
