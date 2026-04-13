"""Advanced KAN model built on efficient-kan with:

* Grid extension schedule [3→5→10→20]
* Spline order k=3, learnable base + spline scales
* Multiplicative interaction nodes (MultKAN-style)
* Auxiliary binary classification head (active ≥ threshold)
* MC-Dropout for uncertainty

The model accepts pre-computed descriptor vectors (Morgan + Mordred
concatenation).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiplicativeLayer(nn.Module):
    """Element-wise gated multiplication layer for feature interactions.

    Given input x of dim d, produces 2d intermediate via a linear, splits
    into two halves, and outputs sigmoid(first) * second (dim d).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        gate, val = h.chunk(2, dim=-1)
        return torch.sigmoid(gate) * val


class BRD4KANModel(nn.Module):
    """Full KAN regression model with optional aux classification head.

    Architecture:
        [input] → MultiplicativeLayer → KAN(layer_widths) → regression head
                                                           ↘ aux classification head
    """

    def __init__(
        self,
        input_dim: int,
        layer_widths: list[int],
        grid_size: int = 3,
        spline_order: int = 3,
        dropout: float = 0.1,
        use_mult_layer: bool = True,
        aux_head: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.dropout_rate = dropout
        self.use_mult_layer = use_mult_layer
        self.has_aux_head = aux_head

        # Optional multiplicative gating
        if use_mult_layer:
            self.mult_layer = MultiplicativeLayer(input_dim)
        else:
            self.mult_layer = nn.Identity()

        # Build KAN layers using efficient_kan
        try:
            from efficient_kan import KANLinear  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "efficient-kan is required. Install with: pip install efficient-kan"
            )

        dims = [input_dim] + layer_widths
        self.kan_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.kan_layers.append(
                KANLinear(
                    dims[i],
                    dims[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                )
            )

        last_dim = layer_widths[-1] if layer_widths else input_dim

        # Regression head
        self.reg_head = nn.Linear(last_dim, 1)

        # Auxiliary binary classification head
        if aux_head:
            self.aux_head = nn.Linear(last_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (pIC50_pred, aux_logits_or_None)."""
        h = self.mult_layer(x)

        for layer in self.kan_layers:
            h = layer(h)
            h = self.dropout(h)

        reg = self.reg_head(h).squeeze(-1)
        aux = self.aux_head(h).squeeze(-1) if self.has_aux_head else None
        return reg, aux

    def regularization_loss(self) -> torch.Tensor:
        """L1 + entropy sparsification over all KAN spline weights."""
        l1 = torch.tensor(0.0, device=next(self.parameters()).device)
        ent = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.kan_layers:
            if hasattr(layer, "scaled_spline_weight"):
                w = layer.scaled_spline_weight
                l1 = l1 + w.abs().mean()
                # Entropy of normalized abs weights
                p = w.abs() / (w.abs().sum() + 1e-12)
                ent = ent - (p * (p + 1e-12).log()).sum()
        return l1, ent  # type: ignore[return-value]

    def update_grid(self, new_grid_size: int) -> None:
        """Extend grid resolution for all KAN layers (progressive refinement)."""
        try:
            from efficient_kan import KANLinear  # type: ignore[import-not-found]
        except ImportError:
            return

        new_layers = nn.ModuleList()
        for layer in self.kan_layers:
            new_layer = KANLinear(
                layer.in_features,
                layer.out_features,
                grid_size=new_grid_size,
                spline_order=self.spline_order,
            )
            new_layer.to(next(self.parameters()).device)
            new_layers.append(new_layer)
        self.kan_layers = new_layers
        self.grid_size = new_grid_size

    def sparsity(self) -> float:
        """Fraction of near-zero spline weights (< 0.01 of max)."""
        total = 0
        near_zero = 0
        for layer in self.kan_layers:
            if hasattr(layer, "scaled_spline_weight"):
                w = layer.scaled_spline_weight.detach().abs()
                threshold = 0.01 * w.max()
                total += w.numel()
                near_zero += (w < threshold).sum().item()
        return near_zero / max(total, 1)


class EnsembleKAN(nn.Module):
    """Deep ensemble of ``n`` BRD4KANModel members for uncertainty."""

    def __init__(self, models: list[BRD4KANModel]) -> None:
        super().__init__()
        self.members = nn.ModuleList(models)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        mc_samples: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mean_pred, epistemic_std, aleatoric_std).

        Combines deep ensemble variance (epistemic) with MC-dropout
        variance within each member (aleatoric proxy).
        """
        all_preds: list[torch.Tensor] = []
        for member in self.members:
            member.eval()
            # MC-Dropout: keep dropout active via training mode for forward passes
            member.dropout.train()
            mc_preds = []
            for _ in range(mc_samples):
                with torch.no_grad():
                    reg, _ = member(x)
                mc_preds.append(reg)
            mc_stack = torch.stack(mc_preds, dim=0)  # (mc, batch)
            all_preds.append(mc_stack.mean(dim=0))

        ensemble_stack = torch.stack(all_preds, dim=0)  # (n_members, batch)
        mean_pred = ensemble_stack.mean(dim=0)
        epistemic_std = ensemble_stack.std(dim=0)

        # MC-dropout variance from last member as aleatoric proxy
        aleatoric_std = mc_stack.std(dim=0)  # type: ignore[possibly-undefined]

        return mean_pred, epistemic_std, aleatoric_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = []
        for member in self.members:
            reg, _ = member(x)
            preds.append(reg)
        return torch.stack(preds).mean(dim=0)
