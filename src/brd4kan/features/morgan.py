"""Morgan ECFP4 fingerprints (deterministic)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _import_rdkit() -> tuple[Any, Any, Any]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    return Chem, AllChem, DataStructs


def morgan_fingerprint(
    smiles: str,
    radius: int,
    n_bits: int,
    use_features: bool = False,
    use_chirality: bool = False,
) -> np.ndarray | None:
    """Return a uint8 ``(n_bits,)`` ECFP fingerprint for ``smiles``, or ``None``."""
    Chem, AllChem, DataStructs = _import_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits,
        useFeatures=use_features,
        useChirality=use_chirality,
    )
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def morgan_matrix(
    smiles_list: list[str],
    radius: int,
    n_bits: int,
    use_features: bool = False,
    use_chirality: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorize a list of SMILES.

    Returns ``(matrix, valid_mask)``: rows where parsing failed are zero-filled
    and the corresponding ``valid_mask`` entry is ``False``.
    """
    n = len(smiles_list)
    matrix = np.zeros((n, n_bits), dtype=np.uint8)
    valid = np.ones(n, dtype=bool)
    for i, smi in enumerate(smiles_list):
        fp = morgan_fingerprint(smi, radius, n_bits, use_features, use_chirality)
        if fp is None:
            valid[i] = False
        else:
            matrix[i] = fp
    return matrix, valid
