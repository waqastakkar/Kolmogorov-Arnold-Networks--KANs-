"""Run manifests — every stage emits one ``manifest.json`` per execution.

The manifest captures:

* git sha + ISO timestamp
* input file signatures (path + size + mtime; sha256 if small enough)
* output file signatures
* full ``params.yaml`` snapshot (so the run is reconstructable)
* seeds (global + any per-stage seeds)
* environment (python, platform, hostname, torch + cuda versions if present)
* wall-clock duration in seconds
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Manifest:
    stage: str
    git_sha: str
    timestamp: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    params_snapshot: dict[str, Any]
    seeds: dict[str, int]
    env: dict[str, str]
    wall_time_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def get_git_sha(default: str = "uncommitted") -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return default


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def env_snapshot() -> dict[str, str]:
    snap: dict[str, str] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    try:
        import torch  # type: ignore[import-not-found]

        snap["torch"] = torch.__version__
        snap["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            snap["cuda_version"] = torch.version.cuda or "unknown"
    except ImportError:
        pass
    return snap


def write_manifest(manifest: Manifest, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "manifest.json"
    out_path.write_text(
        json.dumps(asdict(manifest), indent=2, default=str, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "git_sha.txt").write_text(manifest.git_sha + "\n", encoding="utf-8")
    return out_path
