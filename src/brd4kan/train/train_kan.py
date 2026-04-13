"""KAN training loop with grid extension, sparsification, early stopping.

Handles a single-member training run. The Stage 6 orchestrator calls this
``n_seeds × ensemble_size`` times to build the full deep ensemble.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from brd4kan.models.kan_model import BRD4KANModel

logger = logging.getLogger(__name__)


def _build_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).float()
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)


def _cosine_lr_lambda(epoch: int, total_epochs: int) -> float:
    return 0.5 * (1.0 + math.cos(math.pi * epoch / max(total_epochs, 1)))


def train_single_kan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hparams: dict[str, Any],
    seed: int,
    device: torch.device,
    active_threshold: float = 6.5,
) -> tuple[BRD4KANModel, dict[str, Any]]:
    """Train one KAN member and return the model + training history.

    ``hparams`` keys (all from Optuna or configs/model.yaml):
        layer_widths, grid_size, spline_order, lr, weight_decay, dropout,
        batch_size, lamb, lamb_entropy, lamb_coef, optimizer, epochs,
        grid_schedule, early_stopping_patience, grad_clip.
    """
    from brd4kan.utils.seed import set_global_seed

    set_global_seed(seed)

    layer_widths = hparams.get("layer_widths", [128, 1])
    grid_size = hparams.get("grid_size", 3)
    spline_order = hparams.get("spline_order", 3)
    dropout = hparams.get("dropout", 0.1)
    lr = hparams.get("lr", 1e-3)
    wd = hparams.get("weight_decay", 1e-5)
    batch_size = hparams.get("batch_size", 64)
    lamb = hparams.get("lamb", 1e-3)
    lamb_entropy = hparams.get("lamb_entropy", 2.0)
    lamb_coef = hparams.get("lamb_coef", 0.0)
    optimizer_name = hparams.get("optimizer", "adamw")
    epochs = hparams.get("epochs", 100)
    grid_schedule = hparams.get("grid_schedule", [3, 5, 10, 20])
    patience = hparams.get("early_stopping_patience", 20)
    grad_clip = hparams.get("grad_clip", 1.0)
    use_mult = hparams.get("multiplicative_nodes", True)
    aux_head = hparams.get("aux_classification_head", True)

    model = BRD4KANModel(
        input_dim=X_train.shape[1],
        layer_widths=layer_widths,
        grid_size=grid_size,
        spline_order=spline_order,
        dropout=dropout,
        use_mult_layer=use_mult,
        aux_head=aux_head,
    ).to(device)

    train_loader = _build_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _build_loader(X_val, y_val, batch_size, shuffle=False)

    if optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=5)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: _cosine_lr_lambda(e, epochs)
    )

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    # Grid extension schedule: switch at evenly-spaced epoch fractions
    grid_switch_epochs = []
    if len(grid_schedule) > 1:
        for i, gs in enumerate(grid_schedule[1:], 1):
            grid_switch_epochs.append((int(epochs * i / len(grid_schedule)), gs))

    best_val_rmse = float("inf")
    best_state = None
    wait = 0
    history: dict[str, list[float]] = {"train_rmse": [], "val_rmse": [], "sparsity": []}

    for epoch in range(epochs):
        # Grid extension check
        for switch_epoch, gs in grid_switch_epochs:
            if epoch == switch_epoch:
                logger.info("Grid extension: %d → %d at epoch %d", model.grid_size, gs, epoch)
                model.update_grid(gs)
                # Re-create optimizer for new params
                if optimizer_name == "lbfgs":
                    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=5)
                else:
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda e: _cosine_lr_lambda(e, epochs)
                )

        # --- Train ---
        model.train()
        train_losses: list[float] = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                reg_pred, aux_logits = model(X_batch)
                loss = mse_loss(reg_pred, y_batch)

                # Aux classification loss
                if aux_logits is not None:
                    y_cls = (y_batch >= active_threshold).float()
                    loss = loss + 0.1 * bce_loss(aux_logits, y_cls)

                # Sparsification
                l1, ent = model.regularization_loss()
                loss = loss + lamb * l1 + lamb_entropy * ent

                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                return loss

            if optimizer_name == "lbfgs":
                loss_val = optimizer.step(closure)  # type: ignore[arg-type]
                train_losses.append(float(loss_val))
            else:
                loss_val = closure()
                optimizer.step()
                train_losses.append(float(loss_val))

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_preds: list[np.ndarray] = []
        val_targets: list[np.ndarray] = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                reg_pred, _ = model(X_batch)
                val_preds.append(reg_pred.cpu().numpy())
                val_targets.append(y_batch.numpy())

        val_pred = np.concatenate(val_preds)
        val_true = np.concatenate(val_targets)
        val_rmse = float(np.sqrt(np.mean((val_true - val_pred) ** 2)))
        spars = model.sparsity()

        history["train_rmse"].append(float(np.mean(train_losses)))
        history["val_rmse"].append(val_rmse)
        history["sparsity"].append(spars)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d, best_val_rmse=%.4f", epoch, best_val_rmse)
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    history["best_val_rmse"] = [best_val_rmse]
    return model, history
