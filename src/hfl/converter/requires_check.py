# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Version gating for the Modelfile ``REQUIRES`` instruction (P3-3).

A Modelfile can declare a minimum (or more elaborate) HFL version
with ``REQUIRES``:

    REQUIRES >=0.6.0
    REQUIRES >=0.6.0,<1.0
    REQUIRES ==0.6.1

When ``/api/create`` materialises a Modelfile, the requirement is
checked against the currently running HFL version (``hfl.__version__``).
Failures raise ``RequiresNotSatisfiedError`` which the route turns
into a 400.
"""

from __future__ import annotations

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from hfl import __version__ as HFL_VERSION

__all__ = [
    "RequiresNotSatisfiedError",
    "InvalidRequiresError",
    "check_requires",
    "parse_requires",
]


class InvalidRequiresError(ValueError):
    """Raised when a ``REQUIRES`` spec is malformed."""


class RequiresNotSatisfiedError(ValueError):
    """Raised when the current HFL version does not satisfy the spec."""

    def __init__(self, spec: str, current: str) -> None:
        super().__init__(f"Modelfile requires HFL {spec}, but this server is {current}")
        self.spec = spec
        self.current = current


def parse_requires(spec: str) -> SpecifierSet:
    """Parse a ``REQUIRES`` value into a PEP 440 ``SpecifierSet``.

    Accepts both shapes users write in practice:
    - Bare version, e.g. ``"0.6.0"`` → treated as ``">=0.6.0"``.
    - Full specifier, e.g. ``">=0.6.0,<1.0"``.
    """
    raw = spec.strip()
    if not raw:
        raise InvalidRequiresError("REQUIRES value is empty")

    # Bare version ("0.6") → upgrade to ">=0.6" for user convenience.
    if raw[0].isdigit():
        try:
            Version(raw)
        except InvalidVersion as exc:
            raise InvalidRequiresError(f"invalid version: {raw!r}") from exc
        raw = f">={raw}"

    try:
        return SpecifierSet(raw)
    except InvalidSpecifier as exc:
        raise InvalidRequiresError(f"invalid REQUIRES spec: {raw!r}") from exc


def check_requires(spec: str, current: str = HFL_VERSION) -> None:
    """Validate that ``current`` satisfies ``spec``.

    Passes silently on success; raises ``RequiresNotSatisfiedError`` on
    failure. Malformed specs raise ``InvalidRequiresError``.
    """
    specset = parse_requires(spec)
    try:
        version = Version(current)
    except InvalidVersion as exc:
        raise InvalidRequiresError(f"invalid current version: {current!r}") from exc
    if version not in specset:
        raise RequiresNotSatisfiedError(spec=spec, current=current)
