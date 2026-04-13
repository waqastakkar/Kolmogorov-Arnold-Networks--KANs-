# BRD4-KAN — top-level developer interface.
# Run from a Bash-style shell (Git Bash on Windows).

PY ?= python
UV ?= uv

.PHONY: help setup lint type test smoke repro all clean

help:
	@echo "Targets:"
	@echo "  setup    Create uv env and install pinned deps + pre-commit"
	@echo "  lint     ruff check"
	@echo "  type     mypy"
	@echo "  test     pytest with coverage (>=85%)"
	@echo "  smoke    End-to-end smoke test (Stages 1-10 + report) + DVC dry run"
	@echo "  repro    Run full DVC pipeline (no force)"
	@echo "  all      lint + type + test + repro"
	@echo "  clean    Remove caches (keeps artifacts/)"

setup:
	$(UV) venv --python 3.11
	$(UV) pip install -e ".[dev]"
	$(UV) run pre-commit install

lint:
	$(UV) run ruff check src tests scripts

type:
	$(UV) run mypy src

test:
	$(UV) run pytest

smoke:
	$(UV) run pytest tests/test_smoke_e2e.py -v -x --no-header
	$(UV) run dvc repro --dry

repro:
	$(UV) run dvc repro

all: lint type test repro

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
