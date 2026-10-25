#!/usr/bin/env bash
# pip-audit over every extra: hfl[all] resolved for each platform and Python
# below, not only the core. security.yml's dependency-scan installs the core
# alone, and Pillow 12.3.0's 18 advisories in [tray] and [imagegen] went unseen
# until a scan of the 0.27.1-all image (2026-10-25). Exit 1 on any finding
# that is not excepted below.
#
# Usage: bash scripts/audit_extras.sh   (needs uv and pip-audit on PATH)
set -euo pipefail
cd "$(dirname "$0")/.."

# Excepted, each with why it cannot reach HFL. Re-read them when one goes:
#   PYSEC-2026-2447      diskcache <=5.6.3 unpickles its cache directory. It
#                        comes with llama-cpp-python, and HFL never uses a disk
#                        cache (tests/test_dependency_pins.py keeps it so).
#   PYSEC-2026-3447      setuptools <83: MANIFEST.in excludes when building an
#                        sdist on macOS. Build time only; torch and vllm hold
#                        setuptools at 80.x on Linux.
#   GHSA-xmj3-9gf7-h52w  setuptools <84.1: the legacy wheel-to-egg install on
#                        Windows. HFL never runs it.
IGNORE=(
    --ignore-vuln PYSEC-2026-2447
    --ignore-vuln PYSEC-2026-3447
    --ignore-vuln GHSA-xmj3-9gf7-h52w
)
TARGETS=(
    "3.11 x86_64-unknown-linux-gnu"
    "3.12 x86_64-unknown-linux-gnu"
    "3.12 aarch64-unknown-linux-gnu"
    "3.12 aarch64-apple-darwin"
    "3.12 x86_64-pc-windows-msvc"
)

status=0
for target in "${TARGETS[@]}"; do
    read -r py platform <<< "$target"
    echo "=> hfl[all], Python $py, $platform"
    if ! MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile pyproject.toml --extra all \
        --python-version "$py" --python-platform "$platform" --quiet \
        | pip-audit --strict --no-deps --disable-pip --requirement /dev/stdin "${IGNORE[@]}"; then
        status=1
    fi
done
exit "$status"
