"""Per-stage run directory layout.

Each pipeline stage writes its **canonical artifact** at a fixed,
DVC-trackable path (e.g. ``artifacts/data/raw/brd4_raw.parquet``) and
parks its **manifest + env snapshot** under a sibling timestamped run
directory (``runs/<git-sha>_<utc-timestamp>/``). This gives reproducibility
without forcing DVC to chase moving filenames.
"""

from __future__ import annotations

from pathlib import Path

from brd4kan.utils.manifest import get_git_sha, utc_compact


def make_run_dir(canonical_dir: Path) -> Path:
    """Create and return ``<canonical_dir>/runs/<sha>_<utc>/``."""
    sha = get_git_sha()[:8]
    run_dir = canonical_dir / "runs" / f"{sha}_{utc_compact()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
