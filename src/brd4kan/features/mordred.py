"""Mordred 2D descriptors with variance + correlation filter and z-score scaler.

The fitted scaler (kept columns + per-column mean / std) is saved as a single
joblib pickle so Stage 9 inference can re-apply identical preprocessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def _import_calculator() -> Any:
    from mordred import Calculator, descriptors

    return Calculator, descriptors


def _import_rdkit() -> Any:
    from rdkit import Chem

    return Chem


class MordredFeaturizer:
    """Fit-once, transform-many Mordred 2D pipeline."""

    def __init__(
        self,
        ignore_3d: bool = True,
        variance_threshold: float = 0.0,
        correlation_threshold: float = 0.95,
    ) -> None:
        self.ignore_3d = ignore_3d
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.kept_columns: list[str] | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    # ----- compute -----
    def _compute(self, smiles_list: list[str]) -> pd.DataFrame:
        Calculator, descriptors = _import_calculator()
        Chem = _import_rdkit()
        calc = Calculator(descriptors, ignore_3D=self.ignore_3d)
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        df = calc.pandas(mols, quiet=True)
        return df.apply(pd.to_numeric, errors="coerce")

    # ----- fit -----
    def fit_transform(self, smiles_list: list[str]) -> np.ndarray:
        df = self._compute(smiles_list)
        df = df.dropna(axis=1, how="all")
        variances = df.var(axis=0, skipna=True).fillna(0.0)
        df = df.loc[:, variances > self.variance_threshold]

        corr = df.corr().abs().fillna(0.0)
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        to_drop = [c for c in upper.columns if (upper[c] > self.correlation_threshold).any()]
        df = df.drop(columns=to_drop)

        df = df.fillna(0.0)
        self.kept_columns = list(df.columns)
        self.mean_ = df.mean(axis=0).to_numpy(dtype=np.float64)
        self.std_ = df.std(axis=0).to_numpy(dtype=np.float64) + 1e-12
        return ((df.to_numpy(dtype=np.float64) - self.mean_) / self.std_).astype(np.float32)

    # ----- transform -----
    def transform(self, smiles_list: list[str]) -> np.ndarray:
        if self.kept_columns is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("MordredFeaturizer must be fit before transform.")
        df = self._compute(smiles_list)
        for c in self.kept_columns:
            if c not in df.columns:
                df[c] = 0.0
        df = df[self.kept_columns].fillna(0.0)
        return ((df.to_numpy(dtype=np.float64) - self.mean_) / self.std_).astype(np.float32)

    # ----- persistence -----
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "ignore_3d": self.ignore_3d,
                "variance_threshold": self.variance_threshold,
                "correlation_threshold": self.correlation_threshold,
                "kept_columns": self.kept_columns,
                "mean_": self.mean_,
                "std_": self.std_,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "MordredFeaturizer":
        d = joblib.load(path)
        f = cls(
            ignore_3d=d["ignore_3d"],
            variance_threshold=d["variance_threshold"],
            correlation_threshold=d["correlation_threshold"],
        )
        f.kept_columns = d["kept_columns"]
        f.mean_ = d["mean_"]
        f.std_ = d["std_"]
        return f
