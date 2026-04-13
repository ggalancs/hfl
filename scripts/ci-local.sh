#!/usr/bin/env bash
# Local CI simulation — runs every command from .github/workflows/ci.yml
# in a disposable venv that mirrors the CI environment (no ``[llama]``
# extra, only ``[dev]``).
#
# Why this exists:
#   Every GitHub Actions run consumes paid minutes. We must know the
#   exit code of each job before pushing. This script is the single
#   source of truth for "will CI pass?". If this script exits 0, the
#   push is safe; if it exits non-zero, fix the problem locally first.
#
# Jobs reproduced (see .github/workflows/ci.yml):
#   1. lint       — ``ruff check`` + ``ruff format --check``
#   2. type-check — ``mypy src/hfl/api/ src/hfl/cli/ --ignore-missing-imports``
#   3. test       — ``pytest tests/ --cov=hfl --cov-report=xml``
#
# Matrix note:
#   CI runs tests on Python 3.10/3.11/3.12 × ubuntu/macOS. This script
#   only runs the host Python version. If you want to catch a
#   Python-version regression before pushing, rerun with
#   ``HFL_CI_PY=python3.10 scripts/ci-local.sh`` (etc.) on a machine
#   that has the corresponding interpreter available.
#
# Usage:
#   bash scripts/ci-local.sh            # reuses .venv-ci if it exists
#   CLEAN=1 bash scripts/ci-local.sh    # force-rebuild .venv-ci
#
# Exit codes:
#   0  — every CI step passed; the push is safe
#   1+ — at least one step failed; the output names the offending step

set -euo pipefail

cd "$(dirname "$0")/.."

VENV_DIR="${VENV_DIR:-.venv-ci}"
PYTHON="${HFL_CI_PY:-python3}"

blue() { printf "\033[1;34m%s\033[0m\n" "$*"; }
red() { printf "\033[1;31m%s\033[0m\n" "$*"; }
green() { printf "\033[1;32m%s\033[0m\n" "$*"; }

fail_step() {
    red "✗ $1 failed — fix locally before pushing"
    exit 1
}

# ------------------------------------------------------------------
# 1. Disposable venv
# ------------------------------------------------------------------

if [[ "${CLEAN:-0}" == "1" && -d "$VENV_DIR" ]]; then
    blue "=> Removing stale $VENV_DIR (CLEAN=1)"
    rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
    blue "=> Creating CI venv at $VENV_DIR with $PYTHON"
    "$PYTHON" -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --quiet --upgrade pip
    blue "=> Installing ``pip install -e .[dev]`` (matches CI)"
    pip install --quiet -e ".[dev]"
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    # If pyproject changed since the venv was built, rebuild deps.
    if [[ pyproject.toml -nt "$VENV_DIR/pyvenv.cfg" ]]; then
        blue "=> pyproject.toml is newer than venv; reinstalling"
        pip install --quiet -e ".[dev]"
        touch "$VENV_DIR/pyvenv.cfg"
    fi
fi

# Sanity: ensure llama_cpp is NOT installed in the CI venv so we catch
# test suites that would otherwise fail on CI. The [llama] extra is an
# opt-in optional dep and must not be pulled in here.
if python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('llama_cpp') is None else 1)"; then
    blue "=> llama_cpp not installed (matches CI)"
else
    red "!! llama_cpp is installed in $VENV_DIR — this masks CI test failures"
    red "   Rebuild the venv with CLEAN=1 and do not add [llama] to the install"
    exit 2
fi

# ------------------------------------------------------------------
# 2. Lint (ci.yml ``lint`` job)
# ------------------------------------------------------------------

blue "=> [lint] ruff check src/hfl tests/"
ruff check src/hfl tests/ || fail_step "ruff check"

blue "=> [lint] ruff format --check src/hfl tests/"
ruff format --check src/hfl tests/ || fail_step "ruff format --check"

# ------------------------------------------------------------------
# 3. Type check (ci.yml ``type-check`` job)
# ------------------------------------------------------------------

blue "=> [type-check] mypy src/hfl/api/ src/hfl/cli/ --ignore-missing-imports"
mypy src/hfl/api/ src/hfl/cli/ --ignore-missing-imports || fail_step "mypy"

# ------------------------------------------------------------------
# 4. Test (ci.yml ``test`` job — current Python only)
# ------------------------------------------------------------------

blue "=> [test] pytest tests/ --cov=hfl --cov-report=xml --cov-report=term-missing"
pytest tests/ --cov=hfl --cov-report=xml --cov-report=term-missing || fail_step "pytest"

green ""
green "✓ All CI steps passed locally. Safe to push."
