"""Stage 3 split tests — the headline assertion is *zero scaffold leakage*.

PLAN.md mandates that no Bemis-Murcko scaffold appears in more than one
of {train, val, test}. We assert that property strictly here.
"""

from __future__ import annotations

import pandas as pd
import pytest

rdkit = pytest.importorskip("rdkit")

from brd4kan.data.split import (  # noqa: E402
    bemis_murcko_scaffold,
    scaffold_split,
    time_split,
)


# A small mixed set: several molecules per scaffold so the leakage check is meaningful.
SMILES_SET = [
    # benzene-derived
    "c1ccccc1",
    "c1ccc(O)cc1",
    "c1ccc(N)cc1",
    "c1ccc(Cl)cc1",
    # naphthalene-derived
    "c1ccc2ccccc2c1",
    "c1ccc2cc(O)ccc2c1",
    "c1ccc2cc(N)ccc2c1",
    # pyridine-derived
    "c1ccncc1",
    "c1cc(C)ncc1",
    "c1cc(O)ncc1",
    # quinoline-derived
    "c1ccc2ncccc2c1",
    "c1ccc2nc(C)ccc2c1",
    # piperidine
    "C1CCNCC1",
    "C1CC(C)NCC1",
    # acyclic (no scaffold) — should each get a unique singleton group
    "CCCCCCCC",
    "CC(C)C(=O)O",
    "CCCCC(=O)N",
]


def test_bemis_murcko_groups_share_scaffold() -> None:
    sb = bemis_murcko_scaffold("c1ccccc1")
    sa = bemis_murcko_scaffold("c1ccc(O)cc1")
    assert sb == sa


def test_scaffold_split_zero_leakage() -> None:
    train, val, test = scaffold_split(SMILES_SET, 0.6, 0.2, 0.2)
    train_set = {bemis_murcko_scaffold(SMILES_SET[i]) or f"__singleton_{i}__" for i in train}
    val_set = {bemis_murcko_scaffold(SMILES_SET[i]) or f"__singleton_{i}__" for i in val}
    test_set = {bemis_murcko_scaffold(SMILES_SET[i]) or f"__singleton_{i}__" for i in test}
    assert not (train_set & val_set), f"leak train↔val: {train_set & val_set}"
    assert not (train_set & test_set), f"leak train↔test: {train_set & test_set}"
    assert not (val_set & test_set), f"leak val↔test: {val_set & test_set}"


def test_scaffold_split_partitions_all_indices() -> None:
    train, val, test = scaffold_split(SMILES_SET, 0.6, 0.2, 0.2)
    assert sorted(train + val + test) == list(range(len(SMILES_SET)))
    assert len(set(train + val + test)) == len(SMILES_SET)


def test_scaffold_split_is_deterministic() -> None:
    a = scaffold_split(SMILES_SET, 0.6, 0.2, 0.2)
    b = scaffold_split(SMILES_SET, 0.6, 0.2, 0.2)
    assert a == b


def test_scaffold_split_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError):
        scaffold_split(SMILES_SET, 0.6, 0.2, 0.3)
    with pytest.raises(ValueError):
        scaffold_split(SMILES_SET, -0.1, 0.5, 0.6)


def test_time_split_uses_year_quantile() -> None:
    df = pd.DataFrame(
        {"first_publication_year": [2010, 2012, 2015, 2018, 2019, 2020, 2021, 2022, 2023, 2024]}
    )
    train, test = time_split(df, "first_publication_year", 0.7)
    assert all(df.loc[i, "first_publication_year"] < 2022 for i in train)
    assert all(df.loc[i, "first_publication_year"] >= 2022 for i in test)
    assert sorted(train + test) == list(range(len(df)))
