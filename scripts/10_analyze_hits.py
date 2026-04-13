"""Stage 10 entrypoint -- thin wrapper around ``brd4kan analyze-hits`` for DVC."""

from __future__ import annotations

from brd4kan.cli import app

if __name__ == "__main__":
    app(["analyze-hits"])
