# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Power-source detection for macOS laptops.

Apple Silicon MacBooks clock the GPU down hard on battery. Measured on an
M3 Max (40-core GPU) with Qwen3-14B Q4_K_M, same build, same prompt:

    battery : 113 tok/s prefill,  4.7 tok/s generation
    AC power: 393 tok/s prefill, 38.7 tok/s generation

That is 8x on token generation — far beyond the 30-50% usually quoted, and
invisible from inside the process: the model still reports every layer
offloaded to Metal, it just crawls. Surfacing it turns "why is my Mac slow
at inference?" into a one-line answer.

Everything here is best-effort and non-fatal: an unreadable power state
returns ``None`` ("unknown"), never an exception.
"""

from __future__ import annotations

import logging
import platform
import subprocess

logger = logging.getLogger(__name__)

__all__ = ["on_battery", "power_source_label"]

# ``pmset`` is cheap but still a subprocess; a model load is not a hot path,
# yet a chatty caller shouldn't fork once per request. Cache the first answer
# for the life of the process — a laptop rarely changes power source mid-run,
# and the value is advisory.
_CACHED: bool | None = None
_PROBED = False


def on_battery() -> bool | None:
    """``True`` on battery, ``False`` on AC, ``None`` when undetermined.

    Non-macOS platforms return ``None`` — this is deliberately not a
    general "is this a laptop" probe, only the macOS case where the
    performance impact is large and well documented.
    """
    global _CACHED, _PROBED
    if _PROBED:
        return _CACHED

    _PROBED = True
    _CACHED = None

    if platform.system() != "Darwin":
        return None

    try:
        result = subprocess.run(
            ["pmset", "-g", "ps"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("pmset probe failed", exc_info=True)
        return None

    if result.returncode != 0:
        return None

    # "Now drawing from 'Battery Power'" / "Now drawing from 'AC Power'"
    out = result.stdout.lower()
    if "battery power" in out:
        _CACHED = True
    elif "ac power" in out:
        _CACHED = False
    return _CACHED


def power_source_label() -> str:
    """Human-readable power source for diagnostics output."""
    state = on_battery()
    if state is None:
        return "unknown"
    return "battery" if state else "AC power"
