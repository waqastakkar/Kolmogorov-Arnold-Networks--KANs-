"""Chemprop 2.x D-MPNN baseline wrapper.

Chemprop is trained from SMILES directly (no pre-featurization needed).
This module wraps the Chemprop 2.x Lightning-based API into the same
fit/predict/suggest interface used by the other baselines.

If Chemprop is not installed, all functions raise ``ImportError`` cleanly
so the rest of the baseline suite still runs.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _check_chemprop() -> None:
    try:
        import chemprop  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "chemprop >= 2.0 is required for the D-MPNN baseline. "
            "Install with: pip install chemprop"
        ) from exc


def suggest_chemprop(trial: Any) -> dict[str, Any]:
    return {
        "message_hidden_dim": trial.suggest_categorical("message_hidden_dim", [200, 300, 400]),
        "depth": trial.suggest_int("depth", 2, 5),
        "ffn_hidden_dim": trial.suggest_categorical("ffn_hidden_dim", [200, 300, 400]),
        "ffn_num_layers": trial.suggest_int("ffn_num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 20, 50, step=10),
    }


class ChempropModel:
    """Thin wrapper that stores SMILES + targets, trains, and predicts."""

    def __init__(self, hparams: dict[str, Any], seed: int) -> None:
        _check_chemprop()
        self.hparams = hparams
        self.seed = seed
        self._model_dir: Path | None = None

    def fit(self, smiles: list[str], y: np.ndarray) -> None:
        import chemprop  # type: ignore[import-not-found]
        import lightning as L

        L.seed_everything(self.seed, workers=True)

        data = [chemprop.data.MoleculeDatapoint(chemprop.data.MoleculeDataset.make_mol(s), [float(v)]) for s, v in zip(smiles, y)]

        train_data = chemprop.data.MoleculeDataset(data)
        train_loader = chemprop.data.build_dataloader(train_data, batch_size=self.hparams.get("batch_size", 64), shuffle=True)

        mp = chemprop.nn.BondMessagePassing(
            d_h=self.hparams.get("message_hidden_dim", 300),
            depth=self.hparams.get("depth", 3),
            dropout=self.hparams.get("dropout", 0.0),
        )
        agg = chemprop.nn.MeanAggregation()
        ffn = chemprop.nn.RegressionFFN(
            input_dim=mp.output_dim,
            hidden_dim=self.hparams.get("ffn_hidden_dim", 300),
            n_layers=self.hparams.get("ffn_num_layers", 2),
            dropout=self.hparams.get("dropout", 0.0),
        )
        model = chemprop.models.MPNN(mp, agg, ffn, batch_norm=True)

        trainer = L.Trainer(
            max_epochs=self.hparams.get("epochs", 30),
            accelerator="auto",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            deterministic=True,
        )
        trainer.fit(model, train_loader)

        self._model_dir = Path(tempfile.mkdtemp(prefix="chemprop_"))
        import torch
        torch.save(model.state_dict(), self._model_dir / "model.pt")
        self._model_obj = model

    def predict(self, smiles: list[str]) -> np.ndarray:
        import chemprop  # type: ignore[import-not-found]
        import lightning as L

        data = [chemprop.data.MoleculeDatapoint(chemprop.data.MoleculeDataset.make_mol(s)) for s in smiles]
        dataset = chemprop.data.MoleculeDataset(data)
        loader = chemprop.data.build_dataloader(dataset, batch_size=128, shuffle=False)

        trainer = L.Trainer(
            accelerator="auto",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        preds = trainer.predict(self._model_obj, loader)
        import torch
        return torch.cat(preds).numpy().ravel()

    def save(self, path: Path) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model_obj.state_dict(), path)

    @classmethod
    def load(cls, path: Path, hparams: dict[str, Any], seed: int) -> "ChempropModel":
        obj = cls(hparams, seed)
        _check_chemprop()
        import chemprop  # type: ignore[import-not-found]
        import torch

        mp = chemprop.nn.BondMessagePassing(
            d_h=hparams.get("message_hidden_dim", 300),
            depth=hparams.get("depth", 3),
            dropout=hparams.get("dropout", 0.0),
        )
        agg = chemprop.nn.MeanAggregation()
        ffn = chemprop.nn.RegressionFFN(
            input_dim=mp.output_dim,
            hidden_dim=hparams.get("ffn_hidden_dim", 300),
            n_layers=hparams.get("ffn_num_layers", 2),
            dropout=hparams.get("dropout", 0.0),
        )
        model = chemprop.models.MPNN(mp, agg, ffn, batch_norm=True)
        model.load_state_dict(torch.load(path, weights_only=True))
        obj._model_obj = model
        return obj
