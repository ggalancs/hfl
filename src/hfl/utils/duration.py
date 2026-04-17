# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Ollama-compatible ``keep_alive`` duration parsing.

Ollama's ``keep_alive`` field (present in ``/api/generate``,
``/api/chat`` and ``/api/embed``) accepts:

- Numeric seconds: ``10`` (int or float) → load for 10 seconds.
- Go-style duration string: ``"5m"``, ``"30s"``, ``"1h30m"``, ``"2h"``.
- ``0`` → unload immediately after the request.
- ``-1`` or ``"-1"`` → keep loaded indefinitely (no deadline).
- ``None`` / missing → use the server's default (HFL's pool
  ``idle_timeout_seconds`` continues to govern).

The parser returns one of:

- ``None`` — "no deadline set" (use default idle timeout).
- ``timedelta(seconds=0)`` — "unload after this request".
- ``timedelta(seconds=-1)`` — sentinel for "never expire".
- ``timedelta(...)`` — explicit future deadline.

Calling code is expected to translate the returned timedelta into an
absolute ``datetime`` via ``datetime.now(tz=utc) + delta`` right
before persistence, so the recorded deadline survives server clock
skew debates cleanly.
"""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Union

# Matches Go-style duration components: "1h", "30m", "45s", "500ms",
# "0.5s" (decimals allowed). Ollama's implementation uses Go's
# ``time.ParseDuration`` — we mirror the subset that appears in real
# model-serving traffic.
_COMPONENT_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|us|µs|ns|s|m|h)")
_NUMBER_ONLY_RE = re.compile(r"^-?\d+(?:\.\d+)?$")

# Sentinel for "keep loaded forever". Encoded as timedelta(-1s) so
# callers can branch with ``is NEVER_EXPIRE`` (identity) or with a
# simple ``<= timedelta(0)`` ordering. The value is never added to a
# datetime — callers must check for the sentinel first.
NEVER_EXPIRE = timedelta(seconds=-1)

# Sentinel for "unload immediately". Distinguished from NEVER_EXPIRE
# by a positive-vs-negative check.
UNLOAD_AFTER = timedelta(0)


class InvalidKeepAliveError(ValueError):
    """Raised when ``keep_alive`` cannot be parsed."""


def parse_keep_alive(value: Union[str, int, float, None]) -> timedelta | None:
    """Parse an Ollama-style ``keep_alive`` value.

    Args:
        value: The raw field from the request body. Accepts ``None``,
            numeric seconds, or a Go-style duration string.

    Returns:
        ``None`` when the caller did not supply a value (fall back to
        default idle timeout), ``NEVER_EXPIRE`` for the "-1" / infinite
        case, ``UNLOAD_AFTER`` (``timedelta(0)``) for the "0" /
        immediate-unload case, or a positive ``timedelta`` otherwise.

    Raises:
        InvalidKeepAliveError: The value is a non-empty string that
            doesn't match any known format, or a number outside the
            acceptable range (< -1).
    """
    if value is None:
        return None

    # Already a timedelta (internal callers)
    if isinstance(value, timedelta):
        if value < timedelta(0) and value != NEVER_EXPIRE:
            raise InvalidKeepAliveError(
                f"keep_alive cannot be negative (got {value.total_seconds()}s)"
            )
        return value

    # Numeric (int / float) — treat as raw seconds
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == -1:
            return NEVER_EXPIRE
        if value == 0:
            return UNLOAD_AFTER
        if value < 0:
            raise InvalidKeepAliveError(
                f"keep_alive must be 0, -1, or a positive number (got {value})"
            )
        return timedelta(seconds=float(value))

    # String path
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        # Plain number as string ("5", "0", "-1")
        if _NUMBER_ONLY_RE.match(stripped):
            return parse_keep_alive(float(stripped))
        # Go-style duration: one or more <number><unit> components
        matches = list(_COMPONENT_RE.finditer(stripped))
        if not matches:
            raise InvalidKeepAliveError(
                f"keep_alive {value!r} is not a recognised duration "
                '(use seconds like "5m", "30s", "1h30m", or raw numbers)'
            )
        # Make sure the whole string is consumed by the matches, no
        # stray characters (e.g. ``"5minutes"`` is NOT valid Ollama).
        consumed = sum(m.end() - m.start() for m in matches)
        # Strip optional leading sign for validation
        sign_less = stripped[1:] if stripped.startswith(("+", "-")) else stripped
        if consumed != len(sign_less):
            raise InvalidKeepAliveError(
                f"keep_alive {value!r} has unrecognised trailing characters"
            )
        total_seconds = 0.0
        for m in matches:
            amount, unit = float(m.group(1)), m.group(2)
            total_seconds += _to_seconds(amount, unit)
        if stripped.startswith("-"):
            # Only "-1" makes sense as a negative duration string; any
            # other negative is invalid.
            if total_seconds == 1:
                return NEVER_EXPIRE
            raise InvalidKeepAliveError(f"keep_alive {value!r}: only -1 is a valid negative value")
        if total_seconds == 0:
            return UNLOAD_AFTER
        return timedelta(seconds=total_seconds)

    raise InvalidKeepAliveError(
        f"keep_alive must be a number, string, or None (got {type(value).__name__})"
    )


def _to_seconds(amount: float, unit: str) -> float:
    """Convert a single Go-duration component to seconds."""
    if unit == "ns":
        return amount / 1e9
    if unit in ("us", "µs"):
        return amount / 1e6
    if unit == "ms":
        return amount / 1e3
    if unit == "s":
        return amount
    if unit == "m":
        return amount * 60.0
    if unit == "h":
        return amount * 3600.0
    raise InvalidKeepAliveError(f"Unknown duration unit: {unit}")


def is_never_expire(delta: timedelta | None) -> bool:
    """True iff ``delta`` is the ``NEVER_EXPIRE`` sentinel."""
    return delta is not None and delta == NEVER_EXPIRE


def is_unload_immediately(delta: timedelta | None) -> bool:
    """True iff ``delta`` means "unload right after this request"."""
    return delta is not None and delta == UNLOAD_AFTER
