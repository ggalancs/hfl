#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""llama.cpp's llama-server into a folder, for an executable to bundle.

    python scripts/fetch_llama_server.py build/llama.cpp [--variant cpu]
    HFL_PYI_LLAMA_CPP=build/llama.cpp pyinstaller hfl.spec

The same pinned release, sha256 check and "does it run here" check as
``hfl install llama-server`` (``hfl.engine.llama_server_dist``), so the
executables, the DMG and the MSI carry exactly what a pip install fetches —
on Windows with Microsoft's C++ runtime beside it, which llama.cpp's build
needs and does not ship.
Prints the folder's files; exit 1 if anything failed (nothing is left).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from hfl.engine import llama_server_dist as dist


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--variant", help=f"default: {dist.default_variant()}")
    args = parser.parse_args()
    try:
        server = dist.install(args.variant, target=args.folder.resolve())
    except dist.InstallError as exc:
        print(f"llama-server: {exc}", file=sys.stderr)
        return 1
    if os.name == "nt":
        # llama.cpp's Windows build links Microsoft's C++ runtime and does not
        # ship it (it expects the VC++ Redistributable). Next to the program,
        # Windows loads these first: the bundle runs where it is missing.
        system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
            if not (system / name).is_file():
                print(f"{name} is not in {system}: no VC++ runtime to bundle", file=sys.stderr)
                return 1
            shutil.copy2(system / name, server.parent / name)
    print(dist.check(server).splitlines()[0])
    for path in sorted(server.parent.iterdir()):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
