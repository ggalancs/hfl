# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Metrics endpoints for monitoring.

Provides Prometheus-compatible metrics export.
"""

from typing import Any

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from hfl.metrics import get_metrics

router = APIRouter(tags=["metrics"])


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    tags=["Metrics"],
    summary="Prometheus metrics",
)
async def prometheus_metrics() -> str:
    """Export metrics in Prometheus format.

    Returns metrics suitable for scraping by Prometheus or compatible systems.

    Example output:
        # HELP hfl_requests_total Total number of requests
        # TYPE hfl_requests_total counter
        hfl_requests_total 1234
    """
    return get_metrics().export_prometheus()


@router.get("/metrics/json", tags=["Metrics"], summary="JSON metrics")
async def json_metrics() -> dict[str, Any]:
    """Export metrics as JSON.

    Returns metrics in a structured JSON format for dashboards
    and monitoring systems that prefer JSON over Prometheus format.
    """
    return get_metrics().export_json()
