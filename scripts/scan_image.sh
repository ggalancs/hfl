#!/usr/bin/env bash
# grype over Docker images: every critical or high advisory that has a fix
# fails, except those excepted in scripts/grype-exceptions.yaml (each says
# why it cannot reach HFL). pip-audit (audit_extras.sh) reads only the
# wheels' metadata; an image also carries Debian's packages, Python itself
# and the libraries bundled inside wheels. 0.27.1's image shipped curl with
# critical CVEs that a rebuild would have fixed (found 2026-10-25).
#
# Usage: bash scripts/scan_image.sh IMAGE...   (needs grype on PATH)
#   IMAGE: a local tag, or registry:ghcr.io/ggalancs/hfl:X.Y.Z
# Exit 1 on a finding, 2 when an image could not be scanned.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -eq 0 ]; then
    echo "usage: $0 IMAGE..." >&2
    exit 2
fi

status=0
for image in "$@"; do
    echo "=> $image"
    rc=0
    grype "$image" --only-fixed --fail-on high \
        --config scripts/grype-exceptions.yaml --output table || rc=$?
    case "$rc" in
        0) ;;
        2) echo "FINDINGS in $image: a fixed critical or high advisory" >&2
           [ "$status" -eq 2 ] || status=1 ;;
        *) echo "NOT SCANNED: grype failed on $image (exit $rc)" >&2
           status=2 ;;
    esac
done
exit "$status"
