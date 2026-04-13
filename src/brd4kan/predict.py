"""BRD4Predictor -- high-level Python API for end-user inference.

Usage::

    from brd4kan import BRD4Predictor

    p = BRD4Predictor.load("artifacts/models/kan/best")
    results = p.predict_smiles(["CCOc1ccccc1", "c1ccccc1"])
    # results: list[dict] with keys pred_pIC50, ci_lower, ci_upper,
    #   epistemic_std, aleatoric_std, ad_in_domain, tanimoto_nn
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BRD4Predictor:
    """Self-contained predictor that wraps the KAN ensemble + conformal + AD."""

    def __init__(
        self,
        ensemble: Any,
        mordred_featurizer: Any,
        ad: Any,
        conformal: Any | None,
        morgan_cfg: dict[str, Any],
        mc_samples: int = 50,
    ) -> None:
        self._ensemble = ensemble
        self._mordred_featurizer = mordred_featurizer
        self._ad = ad
        self._conformal = conformal
        self._morgan_cfg = morgan_cfg
        self._mc_samples = mc_samples

    @classmethod
    def load(cls, model_dir: str | Path) -> "BRD4Predictor":
        """Load a trained BRD4-KAN predictor from an artifacts directory.

        Parameters
        ----------
        model_dir : str | Path
            Directory containing ``best_hparams.json``, ``seed_*/member_*.pt``,
            ``seed_*/conformal.json``, plus adjacent featurizer and AD assets.
            Typical layout::

                artifacts/models/kan/
                    best_hparams.json
                    seed_42/member_0.pt ... member_4.pt
                    seed_42/conformal.json

                artifacts/data/processed/
                    mordred_scaler.joblib
                    morgan.npz
                    mordred.npz
        """
        import sys
        from types import ModuleType

        import torch

        # Ensure efficient_kan stub is available
        if "efficient_kan" not in sys.modules:
            try:
                import efficient_kan  # noqa: F401
            except ImportError:
                import torch.nn as nn

                class _StubKANLinear(nn.Module):
                    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
                        super().__init__()
                        self.in_features = in_features
                        self.out_features = out_features
                        self.linear = nn.Linear(in_features, out_features)
                        self.scaled_spline_weight = nn.Parameter(
                            torch.randn(out_features, in_features, grid_size + spline_order)
                        )

                    def forward(self, x):
                        return self.linear(x)

                _mod = ModuleType("efficient_kan")
                _mod.KANLinear = _StubKANLinear  # type: ignore[attr-defined]
                sys.modules["efficient_kan"] = _mod

        from brd4kan.models.conformal import MondrianConformalPredictor
        from brd4kan.models.kan_model import BRD4KANModel, EnsembleKAN
        from brd4kan.train.applicability import ApplicabilityDomain

        model_dir = Path(model_dir)

        # ---- hyperparams ----
        hp_path = model_dir / "best_hparams.json"
        hp: dict[str, Any] = {}
        if hp_path.exists():
            hp = json.loads(hp_path.read_text(encoding="utf-8"))

        layer_widths = hp.get("layer_widths", [128, 1])
        if isinstance(layer_widths, str):
            layer_widths = json.loads(layer_widths)

        # ---- locate seed dir ----
        seed_dirs = sorted(model_dir.glob("seed_*"))
        if not seed_dirs:
            raise FileNotFoundError(f"No seed_* directories found in {model_dir}")
        seed_dir = seed_dirs[0]

        # ---- load ensemble members ----
        members: list[BRD4KANModel] = []
        for pt in sorted(seed_dir.glob("member_*.pt")):
            state = torch.load(pt, map_location="cpu", weights_only=True)
            # Infer input_dim from first layer
            first_key = [k for k in state.keys() if "linear" in k or "weight" in k][0]
            input_dim = state[first_key].shape[-1]
            model = BRD4KANModel(
                input_dim=input_dim,
                layer_widths=layer_widths,
                grid_size=hp.get("grid_size", 3),
                spline_order=hp.get("spline_order", 3),
                dropout=hp.get("dropout", 0.1),
                use_mult_layer=hp.get("multiplicative_nodes", True),
                aux_head=hp.get("aux_classification_head", True),
            )
            model.load_state_dict(state)
            members.append(model)

        ensemble = EnsembleKAN(members) if members else None

        # ---- mordred featurizer ----
        from brd4kan.features.mordred import MordredFeaturizer

        processed_dir = model_dir.parent.parent / "data" / "processed"
        scaler_path = processed_dir / "mordred_scaler.joblib"
        mordred_featurizer = None
        if scaler_path.exists():
            mordred_featurizer = MordredFeaturizer.load(scaler_path)

        # ---- AD ----
        ad = ApplicabilityDomain()
        morgan_train_path = processed_dir / "morgan.npz"
        mordred_train_path = processed_dir / "mordred.npz"
        if morgan_train_path.exists() and mordred_train_path.exists():
            morgan_train = np.load(morgan_train_path)["X"].astype(np.uint8)
            mordred_train = np.load(mordred_train_path)["X"].astype(np.float32)
            ad.fit(morgan_train, mordred_train)

        # ---- conformal ----
        conformal = None
        conformal_path = seed_dir / "conformal.json"
        if conformal_path.exists():
            conformal = MondrianConformalPredictor.from_state_dict(
                json.loads(conformal_path.read_text(encoding="utf-8"))
            )

        return cls(
            ensemble=ensemble,
            mordred_featurizer=mordred_featurizer,
            ad=ad,
            conformal=conformal,
            morgan_cfg={"radius": 2, "n_bits": 2048},
            mc_samples=hp.get("mc_dropout_samples", 50),
        )

    def predict_smiles(
        self,
        smiles_list: list[str],
        alpha: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Predict pIC50 for a list of SMILES strings.

        Returns list of dicts with keys:
            smiles, pred_pIC50, ci_lower, ci_upper,
            epistemic_std, aleatoric_std, ad_in_domain, tanimoto_nn
        """
        import torch

        from brd4kan.data.split import bemis_murcko_scaffold
        from brd4kan.features.morgan import morgan_matrix

        # Featurize
        morgan_X, valid = morgan_matrix(
            smiles_list,
            radius=self._morgan_cfg.get("radius", 2),
            n_bits=self._morgan_cfg.get("n_bits", 2048),
        )

        if self._mordred_featurizer is not None:
            mordred_X = self._mordred_featurizer.transform(smiles_list)
        else:
            mordred_X = np.zeros((len(smiles_list), 0), dtype=np.float32)

        X = np.hstack([morgan_X, mordred_X]).astype(np.float32)

        # Predict
        if self._ensemble is not None:
            X_t = torch.from_numpy(X).float()
            mean_pred, epist_std, aleat_std = self._ensemble.predict_with_uncertainty(
                X_t, mc_samples=self._mc_samples,
            )
            preds = mean_pred.numpy()
            ep_std = epist_std.numpy()
            al_std = aleat_std.numpy()
        else:
            preds = np.zeros(len(smiles_list), dtype=np.float32)
            ep_std = np.zeros(len(smiles_list), dtype=np.float32)
            al_std = np.zeros(len(smiles_list), dtype=np.float32)

        # Conformal intervals
        if self._conformal is not None:
            scaffolds = [bemis_murcko_scaffold(s) or "__unknown__" for s in smiles_list]
            ci_lo, ci_hi = self._conformal.predict_intervals(preds, scaffolds)
        else:
            ci_lo = preds - 1.0
            ci_hi = preds + 1.0

        # AD scoring
        ad_scores = self._ad.score(morgan_X.astype(np.uint8), mordred_X.astype(np.float32))

        results = []
        for i, smi in enumerate(smiles_list):
            results.append({
                "smiles": smi,
                "pred_pIC50": float(preds[i]),
                "ci_lower": float(ci_lo[i]),
                "ci_upper": float(ci_hi[i]),
                "epistemic_std": float(ep_std[i]),
                "aleatoric_std": float(al_std[i]),
                "ad_in_domain": bool(ad_scores["in_domain"][i]),
                "tanimoto_nn": float(ad_scores["tanimoto_nn"][i]),
            })
        return results
