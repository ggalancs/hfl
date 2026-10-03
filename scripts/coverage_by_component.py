#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Test coverage per component, each held to a floor.

    pytest --cov=hfl --cov-report=json:coverage.json
    python scripts/coverage_by_component.py coverage.json [--floor 90]

A component is a package under ``src/hfl`` (``api``, ``engine``, ``hub``…;
``cli/commands`` apart from ``cli``); top-level modules are ``(top)``.
The total alone hid components far below it: a large, well-tested package
lifted the average over a small one at 55 %. Exit 1 when any component is
under the floor, naming each with its files that miss the most lines.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def component(path: str) -> str:
    parts = Path(path).parts
    i = parts.index("hfl")
    rest = parts[i + 1 :]
    if len(rest) == 1:
        return "(top)"
    if rest[0] == "cli" and len(rest) > 2:
        return f"cli/{rest[1]}"
    return rest[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report", type=Path, help="coverage JSON (--cov-report=json)")
    parser.add_argument("--floor", type=float, default=90.0)
    args = parser.parse_args()
    files = json.loads(args.report.read_text())["files"]
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    worst: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for path, data in files.items():
        summary = data["summary"]
        # Lines and branches, as coverage's own percentage counts them.
        covered = summary["covered_lines"] + summary.get("covered_branches", 0)
        total = summary["num_statements"] + summary.get("num_branches", 0)
        key = component(path)
        totals[key][0] += covered
        totals[key][1] += total
        missing = total - covered
        if missing:
            worst[key].append((missing, path.split("src/")[-1]))
    failed = []
    for key, (covered, total) in sorted(totals.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1)):
        percent = 100.0 * covered / max(total, 1)
        bad = percent < args.floor
        print(f"{'BAD' if bad else 'OK '} {key:16} {percent:5.1f}%  ({total - covered} of {total} missing)")
        if bad:
            failed.append(key)
            for missing, name in sorted(worst[key], reverse=True)[:3]:
                print(f"      {missing:5d} missing  {name}")
    if failed:
        print(f"under {args.floor:.0f}%: {', '.join(failed)}")
        return 1
    print(f"every component at {args.floor:.0f}% or more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
