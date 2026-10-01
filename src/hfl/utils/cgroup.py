# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The memory a Linux container may use, when it has a limit.

psutil reads the host's memory, not the container's: on Modal a container
limited to 48 GB saw 339.6 GB, so ``hfl pull`` judged that a 235B model
fit split between the GPU and "RAM", downloaded 512 GB and filled the
disk; the residency planner would likewise admit models past the limit,
until the kernel killed the container. A cgroup's limit (``memory.max``,
or ``memory.limit_in_bytes`` in cgroup v1) and usage are what count there.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/sys/fs/cgroup")
_UNLIMITED = 1 << 60  # cgroup v1 writes "no limit" as a huge number


def _on_linux() -> bool:
    return sys.platform.startswith("linux")


def _own_dirs() -> list[Path]:
    """This process's cgroup directories, most specific first: its own
    (from /proc/self/cgroup) and the mount root (a container's view)."""
    dirs = []
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[1] in ("", "memory") and parts[2] not in ("", "/"):
                rel = parts[2].lstrip("/")
                dirs += [ROOT / rel, ROOT / "memory" / rel]
    except OSError:
        pass
    return [*dirs, ROOT, ROOT / "memory"]


def _read_int(directory: Path, *names: str) -> int | None:
    for name in names:
        try:
            raw = (directory / name).read_text().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        return value if value < _UNLIMITED else None
    return None


def limit_bytes() -> int | None:
    """The cgroup's memory limit in bytes; None without one (or off Linux)."""
    if not _on_linux():
        return None
    for directory in _own_dirs():
        limit = _read_int(directory, "memory.max", "memory.limit_in_bytes")
        if limit:
            return limit
    return None


def in_use_bytes() -> int | None:
    """What the cgroup holds, less the page cache it can drop (the same
    sense as psutil's total minus available); None when unreadable."""
    if not _on_linux():
        return None
    for directory in _own_dirs():
        used = _read_int(directory, "memory.current", "memory.usage_in_bytes")
        if used is None:
            continue
        reclaimable = 0
        try:
            for line in (directory / "memory.stat").read_text().splitlines():
                key, _, value = line.partition(" ")
                if key in ("inactive_file", "total_inactive_file"):
                    reclaimable = int(value)
                    break
        except (OSError, ValueError):
            pass
        return max(0, used - reclaimable)
    return None
