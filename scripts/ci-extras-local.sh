#!/usr/bin/env bash
# Local replica of .github/workflows/test-extras.yml: the suite with the
# [llama] and [structured] extras, on Linux (python 3.12), in a container,
# so it costs no Actions minutes. The checkout is copied in (an editable
# install writes into it); nothing is written back.
#
#   bash scripts/ci-extras-local.sh                 # the whole job
#   bash scripts/ci-extras-local.sh tests/test_x.py  # some tests, full output
set -euo pipefail
cd "$(dirname "$0")/.."
image=python:3.12-slim-bookworm
# linux/amd64: the architecture of the GitHub runner, and of the CPU wheels.
docker run --rm --platform linux/amd64 -v "$PWD":/repo:ro -e "ONLY=$*" "$image" bash -c '
  set -euo pipefail
  apt-get update -qq >/dev/null && apt-get install -y -qq --no-install-recommends git >/dev/null
  # The project'"'"'s files only (tracked, and new ones not ignored): not the
  # venvs, nor coverage files a test run on the host may be rewriting.
  git config --global --add safe.directory /repo
  work=$(mktemp -d)
  (cd /repo && git ls-files -co --exclude-standard -z | tar cf - --null -T -) | tar xf - -C "$work"
  cd "$work"
  python -m pip install -q --upgrade pip
  pip install -q --only-binary=llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -e ".[dev,structured,llama,transformers]"
  if [ -n "$ONLY" ]; then
    pytest -q -p no:cacheprovider --no-cov $ONLY
  else
    pytest -q --cov=hfl --cov-report=term-missing:skip-covered --cov-fail-under=80 -p no:cacheprovider
    python -m coverage report --include="src/hfl/engine/constrained.py,src/hfl/engine/llama_cpp.py"
  fi
'
