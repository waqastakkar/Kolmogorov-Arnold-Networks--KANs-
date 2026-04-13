"""Global seeding for full reproducibility (random / numpy / torch / hash / cuBLAS)."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed every RNG used by the project.

    Honors ``BRD4KAN_DETERMINISTIC=1`` to opt into ``torch.use_deterministic_algorithms``
    and the cuBLAS workspace config required for full GPU determinism.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if os.environ.get("BRD4KAN_DETERMINISTIC", "1") == "1":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older torch versions silently lack this; safe to ignore.
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
