"""Stage 3 entrypoint — thin wrapper around ``brd4kan split`` for DVC."""

from __future__ import annotations

from brd4kan.cli import app

if __name__ == "__main__":
    app(["split"])
