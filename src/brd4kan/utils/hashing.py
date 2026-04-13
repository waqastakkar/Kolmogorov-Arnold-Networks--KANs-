"""Content-addressable hashing utilities for manifests and reproducibility checks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

_CHUNK = 1 << 20  # 1 MiB
_DEFAULT_MAX_HASH_BYTES = 200 << 20  # 200 MiB


def file_sha256(path: Path) -> str:
    """Stream-hash a file with SHA-256."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for buf in iter(lambda: fh.read(_CHUNK), b""):
            h.update(buf)
    return h.hexdigest()


def array_sha256(arr: np.ndarray) -> str:
    """Hash a NumPy array's bytes (after C-contiguous canonicalization)."""
    contiguous = np.ascontiguousarray(arr)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def file_signature(path: Path, max_hash_bytes: int = _DEFAULT_MAX_HASH_BYTES) -> dict[str, Any]:
    """Return a manifest-friendly signature for ``path``.

    Hashes small files in full; for large files (e.g., the multi-GB ChEMBL DB),
    records size + mtime so manifests stay fast.
    """
    if not path.exists():
        return {"path": str(path), "status": "missing"}
    st = path.stat()
    sig: dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }
    if st.st_size <= max_hash_bytes:
        sig["sha256"] = file_sha256(path)
    return sig
