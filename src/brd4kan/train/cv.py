"""Scaffold-aware cross-validation inside the training set.

Used by Optuna objectives to score each trial: the train split is further
divided into ``cv_folds`` sub-splits via Bemis-Murcko scaffolds, so no
scaffold appears in both train-fold and val-fold.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def scaffold_cv_indices(
    smiles: Sequence[str],
    n_folds: int,
) -> list[tuple[list[int], list[int]]]:
    """Return ``(train_idx, val_idx)`` per fold — scaffold-grouped k-fold."""
    from brd4kan.data.split import bemis_murcko_scaffold

    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        s = bemis_murcko_scaffold(smi) or f"__singleton_{i}__"
        groups[s].append(i)

    sorted_groups = sorted(groups.values(), key=lambda g: (-len(g), g[0]))

    fold_assignment: list[int] = [-1] * len(smiles)
    fold_sizes = [0] * n_folds
    for g in sorted_groups:
        target = int(np.argmin(fold_sizes))
        for idx in g:
            fold_assignment[idx] = target
        fold_sizes[target] += len(g)

    folds: list[tuple[list[int], list[int]]] = []
    for f in range(n_folds):
        val_idx = [i for i, fa in enumerate(fold_assignment) if fa == f]
        train_idx = [i for i, fa in enumerate(fold_assignment) if fa != f]
        folds.append((train_idx, val_idx))
    return folds
