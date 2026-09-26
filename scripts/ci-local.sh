#!/usr/bin/env bash
# Local CI simulation — runs every gating command from ci.yml, lint.yml and
# test.yml in a disposable venv that mirrors the CI environment (no ``[llama]``
# extra, only ``[dev]``).
#
# Why this exists:
#   Every GitHub Actions run consumes paid minutes. We must know the
#   exit code of each job before pushing. This script is the single
#   source of truth for "will CI pass?". If this script exits 0, the
#   push is safe; if it exits non-zero, fix the problem locally first.
#
# Jobs reproduced — THREE workflows gate a push, not one:
#   ci.yml    lint       — ``ruff check`` + ``ruff format --check``
#             type-check — ``mypy src/hfl/api/ src/hfl/cli/``
#             test       — ``pytest tests/ --cov=hfl``   (floor 75, pyproject)
#   lint.yml  ruff       — same two commands over ``src/`` (WIDER than ci.yml's
#                          ``src/hfl``: a second package under src/ is linted
#                          by lint.yml and invisible to ci.yml)
#             mypy       — ``mypy src/hfl`` (whole package). lint.yml marks its
#                          own job ``continue-on-error: true``, but
#                          tests/test_static_analysis.py runs the same command
#                          inside the suite, so it gates through ci.yml's test
#                          job regardless. Checked early here to fail in
#                          seconds rather than after the full suite.
#   test.yml  test       — same tests with ``--cov-fail-under=80``
#
# The coverage floors differ (75 in ci.yml via pyproject, 80 in test.yml), so
# the suite runs ONCE at the stricter 80: passing 80 passes 75, and a second
# full run would cost another ~85 s to prove nothing.
#
# Deliberately NOT reproduced, because each needs tooling this script will not
# install for you — run them yourself before a release:
#   security.yml       CodeQL (~1 GB bundle), gitleaks, pip-audit --strict
#   license-check.yml  pip-licenses --fail-on="GPL;AGPL;LGPL;..."
#   the matrix         CI covers Python 3.10/3.11/3.12 x ubuntu/macOS; this
#                      script runs one interpreter on this machine (see below)
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
# 2. Lint (ci.yml ``lint`` job + lint.yml ``ruff`` job)
# ------------------------------------------------------------------
# ``src/`` is lint.yml's scope and a superset of ci.yml's ``src/hfl``, so one
# pass satisfies both. Do not narrow it back to ``src/hfl``: that is exactly
# the gap that let lint.yml fail on a tree this script called green.

blue "=> [lint] ruff check src/ tests/ audit/"
ruff check src/ tests/ audit/ || fail_step "ruff check"

blue "=> [lint] ruff format --check src/ tests/ audit/"
ruff format --check src/ tests/ audit/ || fail_step "ruff format --check"

# ------------------------------------------------------------------
# 3. Type check (ci.yml ``type-check`` job — GATING)
# ------------------------------------------------------------------
# Run ci.yml's exact invocation rather than assuming the wider ``mypy src/hfl``
# below subsumes it: mypy's per-module strictness and its module discovery both
# depend on which roots it is given, so the two are not interchangeable.

blue "=> [type-check] mypy src/hfl/api/ src/hfl/cli/ --ignore-missing-imports"
mypy src/hfl/api/ src/hfl/cli/ --ignore-missing-imports || fail_step "mypy"

# ------------------------------------------------------------------
# 3b. Whole-package mypy + ruff — fast-fail for what step 4 enforces anyway
# ------------------------------------------------------------------
# lint.yml's own ``mypy`` job is ``continue-on-error: true``, so it is tempting
# to treat this as advisory. It is not: ``tests/test_static_analysis.py`` shells
# out to ``mypy src/hfl``, ``ruff check src/hfl`` and ``ruff format --check
# src/hfl`` from inside the suite, and the suite IS gating. A type error outside
# ``api/``/``cli/`` therefore sails past step 3 and then fails ci.yml's test job.
#
# Running it here is not extra strictness — it is the same gate, ~85 s earlier
# and as one readable mypy report instead of an assertion buried in pytest
# output. Verified by sabotage: an annotation error in src/hfl/metrics.py is
# invisible to step 3 and fails test_mypy_clean_on_src.
#
# NOTE for releases: this covers only the venv it runs in, the ``[llama]``-ABSENT
# one. The ``[llama]``-present venv fails differently — run ``mypy src/hfl`` in
# .venv too before tagging.

blue "=> [type-check] mypy src/hfl --ignore-missing-imports (whole package)"
mypy src/hfl --ignore-missing-imports || fail_step "mypy src/hfl (test_static_analysis gate)"

# ------------------------------------------------------------------
# 4. Test (ci.yml + test.yml ``test`` jobs — current Python only)
# ------------------------------------------------------------------

blue "=> [test] pytest tests/ --cov=hfl --cov-fail-under=80 (test.yml's floor)"
pytest tests/ --cov=hfl --cov-report=xml --cov-report=term-missing \
    --cov-fail-under=80 || fail_step "pytest"

green ""
green "✓ ci.yml + lint.yml + test.yml all pass locally. Safe to push."
blue  "  Not covered here: security.yml, license-check.yml, and the"
blue  "  3.10/3.11/3.12 x ubuntu/macOS matrix. See the header."
