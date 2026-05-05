# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""V4 F8 — ``GET /api/compliance/dashboard``.

Aggregates the local registry's metadata into a single envelope that
Ollama cannot produce: per-license-risk breakdown, gated repos
without ``HF_TOKEN``, models with no declared license, EU AI Act
warnings. The HF Hub publishes the underlying metadata; HFL turns
it into operator-actionable signals.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from fastapi import APIRouter

from hfl.hub.license_checker import LICENSE_CLASSIFICATION, LicenseRisk

logger = logging.getLogger(__name__)

router = APIRouter(tags=["HFL Beyond"])


def _risk_for_license(license_id: str | None) -> str:
    """Map a license string to the LicenseRisk enum value, lower-cased.

    Unknown / missing → ``"unknown"``. Comparison is case-insensitive
    because the Hub mixes cases (``Apache-2.0`` vs ``apache-2.0``).
    """
    if not license_id:
        return LicenseRisk.UNKNOWN.value
    risk = LICENSE_CLASSIFICATION.get(license_id.strip().lower(), LicenseRisk.UNKNOWN)
    return risk.value


def _build_compliance_dashboard() -> dict[str, Any]:
    """Walk the registry once and tally compliance signals.

    The walk is best-effort: any registry/manifest field we can't
    read becomes an ``unknown`` bucket entry rather than crashing
    the dashboard. Operators want the report even when the registry
    is partially corrupt.
    """
    from hfl.config import config
    from hfl.core.container import get_registry

    try:
        manifests = list(get_registry().list_all())
    except Exception:
        logger.exception("registry walk failed during compliance dashboard")
        manifests = []

    by_risk: Counter[str] = Counter()
    by_license: Counter[str] = Counter()
    gated_without_token: list[str] = []
    missing_license: list[str] = []
    eu_ai_act_warnings: list[dict[str, str]] = []

    has_hf_token = bool(config.hf_token)

    for manifest in manifests:
        license_id = getattr(manifest, "license", None) or getattr(manifest, "license_id", None)
        risk = _risk_for_license(license_id)
        by_risk[risk] += 1
        by_license[license_id or "unknown"] += 1

        if not license_id:
            missing_license.append(manifest.name)

        if getattr(manifest, "gated", False) and not has_hf_token:
            gated_without_token.append(manifest.name)

        # EU AI Act: surface high-risk uses (LLMs serving > N users
        # daily) — we don't have that signal, so we flag based on
        # license families that imply commercial deployment is gated
        # by the upstream provider.
        if risk in (LicenseRisk.NON_COMMERCIAL.value, LicenseRisk.RESTRICTED.value):
            eu_ai_act_warnings.append(
                {
                    "model": manifest.name,
                    "license": license_id or "unknown",
                    "reason": (
                        "Restricted license — verify EU AI Act / commercial "
                        "deployment compatibility before serving externally."
                    ),
                }
            )

    return {
        "total_models": len(manifests),
        "by_risk": dict(by_risk),
        "by_license": dict(by_license),
        "gated_without_token": gated_without_token,
        "missing_license": missing_license,
        "eu_ai_act_warnings": eu_ai_act_warnings,
        "has_hf_token": has_hf_token,
    }


@router.get(
    "/api/compliance/dashboard",
    response_model=None,
    summary="Local-registry compliance overview (V4 F8)",
    responses={200: {"description": "Compliance dashboard"}},
)
async def api_compliance_dashboard() -> dict[str, Any]:
    """Return a one-shot snapshot of compliance-relevant metadata.

    Output shape::

        {
            "total_models": 12,
            "by_risk": {"permissive": 4, "conditional": 6, "unknown": 2},
            "by_license": {"apache-2.0": 4, "llama3.1": 5, ...},
            "gated_without_token": ["meta-llama/foo", ...],
            "missing_license": ["user/some-fork"],
            "eu_ai_act_warnings": [...],
            "has_hf_token": false
        }

    No mutation: this endpoint never writes to the registry.
    """
    return _build_compliance_dashboard()
