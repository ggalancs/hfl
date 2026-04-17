# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Process-level sandboxing hooks (Phase 18 P3 — V2 row 37).

Optional hardening for ``hfl serve``. When the operator passes
``--sandbox <mode>`` (or sets ``HFL_SANDBOX=<mode>``), the server
tries to restrict itself at startup:

- ``seccomp``: Linux-only. Uses ``prctl(PR_SET_NO_NEW_PRIVS)`` +
  a minimal seccomp-bpf filter that blocks ``ptrace`` / ``kexec_*``
  / ``reboot`` / ``unshare`` / ``setuid``. Further tightening can
  come from ``landlock`` when available.
- ``macos``: macOS-only. Emits an App-Sandbox profile hint in the
  log — actual enforcement requires the binary to be codesigned
  with the ``com.apple.security.app-sandbox`` entitlement, which
  the DMG pipeline (V2 row 31) does.
- ``none`` / unset: no-op.

The whole module is defensive: any failure to apply the restriction
logs a WARNING and continues. The intent is "opt-in hardening",
not "hard-fail if unsupported".
"""

from __future__ import annotations

import ctypes
import logging
import platform
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "SandboxResult",
    "apply_sandbox",
    "SUPPORTED_MODES",
]

SUPPORTED_MODES = ("none", "seccomp", "macos")


@dataclass
class SandboxResult:
    """Outcome of ``apply_sandbox`` — inspectable by health checks."""

    mode: str = "none"
    applied: bool = False
    reason: str = "disabled"


# ----------------------------------------------------------------------
# seccomp / Linux
# ----------------------------------------------------------------------


def _apply_linux_seccomp() -> SandboxResult:
    if platform.system() != "Linux":
        return SandboxResult(mode="seccomp", applied=False, reason="not-linux")
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError as exc:
        return SandboxResult(mode="seccomp", applied=False, reason=f"no-libc: {exc}")

    # prctl(PR_SET_NO_NEW_PRIVS=38, 1) is the gate for seccomp in
    # user-space processes — applying it is a no-cost, reversible
    # step that we always try first.
    PR_SET_NO_NEW_PRIVS = 38
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        return SandboxResult(
            mode="seccomp",
            applied=False,
            reason=f"prctl failed ({ctypes.get_errno()})",
        )

    # Real seccomp filter installation requires libseccomp. If
    # pyseccomp is present, use it — otherwise we only set
    # NO_NEW_PRIVS and log partial success.
    try:
        import seccomp  # type: ignore
    except ImportError:
        return SandboxResult(
            mode="seccomp",
            applied=True,
            reason="no-seccomp-library (NO_NEW_PRIVS only)",
        )

    try:
        filter_ = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
        for syscall in (
            "ptrace",
            "kexec_load",
            "kexec_file_load",
            "reboot",
            "unshare",
            "setuid",
            "setgid",
            "seteuid",
            "setegid",
        ):
            filter_.add_rule(seccomp.ERRNO(1), syscall)
        filter_.load()
    except Exception as exc:  # pragma: no cover — defensive
        return SandboxResult(
            mode="seccomp",
            applied=False,
            reason=f"seccomp filter load failed: {exc}",
        )

    return SandboxResult(mode="seccomp", applied=True, reason="filter loaded")


# ----------------------------------------------------------------------
# macOS / App Sandbox hint
# ----------------------------------------------------------------------


def _apply_macos_hint() -> SandboxResult:
    if platform.system() != "Darwin":
        return SandboxResult(mode="macos", applied=False, reason="not-darwin")
    logger.info(
        "macOS sandbox requested: enforcement comes from the signed DMG's "
        "com.apple.security.app-sandbox entitlement. This is an advisory hint."
    )
    return SandboxResult(mode="macos", applied=True, reason="advisory")


# ----------------------------------------------------------------------
# Public entrypoint
# ----------------------------------------------------------------------


def apply_sandbox(mode: str | None) -> SandboxResult:
    """Apply the requested sandbox mode; never raises.

    ``mode`` is case-insensitive. Unknown values fall through to
    ``"none"`` with a warning. The returned ``SandboxResult`` is
    inspectable from ``/healthz/ready`` for operators auditing the
    runtime posture.
    """
    if not mode:
        return SandboxResult()
    normalised = mode.lower().strip()
    if normalised not in SUPPORTED_MODES:
        logger.warning("Unknown --sandbox value %r, ignoring", mode)
        return SandboxResult(mode="none", applied=False, reason=f"unknown mode {mode!r}")
    if normalised == "none":
        return SandboxResult()
    if normalised == "seccomp":
        result = _apply_linux_seccomp()
    elif normalised == "macos":
        result = _apply_macos_hint()
    else:  # pragma: no cover — SUPPORTED_MODES gate
        return SandboxResult(mode="none", applied=False, reason="internal")
    if not result.applied:
        logger.warning("Sandbox %s not applied: %s", result.mode, result.reason)
    else:
        logger.info("Sandbox %s applied (%s)", result.mode, result.reason)
    return result
