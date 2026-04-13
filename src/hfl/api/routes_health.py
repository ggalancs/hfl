# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Health check endpoints for monitoring and orchestration.

Provides:
- /health - Basic health check for load balancers
- /health/ready - Readiness check (can accept requests)
- /health/live - Liveness check (process is alive)
- /health/deep - Deep health check with system metrics
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from hfl.api.state import get_state
from hfl.config import config
from hfl.models.registry import get_registry

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])

_startup_time = datetime.now()


def _check_registry() -> bool:
    """Check if registry is accessible."""
    try:
        get_registry()
        return True
    except (OSError, ValueError, KeyError, AttributeError) as e:
        logger.debug("Registry check failed: %s", e)
        return False


@router.get("/healthz", tags=["Health"], summary="Orchestrator health check")
async def healthz() -> JSONResponse:
    """Orchestrator-friendly health endpoint (spec §5.5).

    Returns 200 with ``status=ok`` when an LLM engine is loaded and ready,
    otherwise 503 with ``status=degraded``. The body always includes the
    list of loaded model names, queue depth, and uptime so operators can
    monitor without scraping multiple endpoints.
    """
    state = get_state()
    uptime = (datetime.now() - _startup_time).total_seconds()

    models_loaded: list[str] = []
    if state.current_model is not None:
        models_loaded.append(state.current_model.name)
    if state.current_tts_model is not None:
        models_loaded.append(state.current_tts_model.name)

    # Live queue_depth from the inference dispatcher (spec §5.3).
    queue_depth = 0
    queue_in_flight = 0
    try:
        from hfl.core import get_dispatcher

        snap = get_dispatcher().snapshot()
        queue_depth = snap.depth
        queue_in_flight = snap.in_flight
    except Exception:  # pragma: no cover — defensive
        pass

    healthy = state.is_llm_loaded() or not models_loaded
    body: dict[str, Any] = {
        "status": "ok" if healthy else "degraded",
        "models_loaded": models_loaded,
        "queue_depth": queue_depth,
        "queue_in_flight": queue_in_flight,
        "uptime_seconds": uptime,
    }
    return JSONResponse(status_code=200 if healthy else 503, content=body)


@router.get("/health", tags=["Health"], summary="Basic health check")
async def health_basic() -> dict[str, Any]:
    """Basic health check - maintains backwards compatibility."""
    state = get_state()
    return {
        "status": "healthy",
        "model_loaded": state.is_llm_loaded(),
        "current_model": state.current_model.name if state.current_model else None,
        "tts_model_loaded": state.is_tts_loaded(),
        "current_tts_model": state.current_tts_model.name if state.current_tts_model else None,
    }


@router.get("/health/ready", tags=["Health"], summary="Readiness probe")
async def health_ready() -> dict[str, Any]:
    """Readiness check - can accept requests.

    Returns ready if:
    - Configuration is loaded
    - Registry is accessible
    """
    state = get_state()

    checks = {
        "config_loaded": config is not None,
        "registry_accessible": _check_registry(),
        "model_loaded": state.is_llm_loaded(),
    }

    is_ready = checks["config_loaded"] and checks["registry_accessible"]

    return {
        "status": "ready" if is_ready else "not_ready",
        "checks": checks,
        "model": state.current_model.name if state.current_model else None,
    }


@router.get("/health/live", tags=["Health"], summary="Liveness probe")
async def health_live() -> dict[str, Any]:
    """Liveness check - process is alive."""
    uptime = (datetime.now() - _startup_time).total_seconds()

    result: dict[str, Any] = {
        "status": "alive",
        "uptime_seconds": uptime,
    }

    # Optional: add memory info if psutil is available
    try:
        import psutil

        result["memory_mb"] = psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        pass

    return result


@router.get("/health/deep", tags=["Health"], summary="Deep health check")
async def health_deep(probe: bool = False) -> dict[str, Any]:
    """Deep health check - all systems.

    Args:
        probe: If True, run a minimal inference test to verify model health.
    """
    state = get_state()
    uptime = (datetime.now() - _startup_time).total_seconds()

    result: dict[str, Any] = {
        "status": "healthy",
        "version": "0.1.0",
        "uptime_seconds": uptime,
        "llm": {
            "loaded": state.is_llm_loaded(),
            "model": state.current_model.name if state.current_model else None,
        },
        "tts": {
            "loaded": state.is_tts_loaded(),
            "model": state.current_tts_model.name if state.current_tts_model else None,
        },
    }

    # Optional inference probe
    if probe and state.is_llm_loaded() and state.engine is not None:
        try:
            import asyncio

            from hfl.engine.base import GenerationConfig

            probe_config = GenerationConfig(max_tokens=1)
            probe_result = await asyncio.to_thread(state.engine.generate, "test", probe_config)
            result["llm"]["probe"] = "ok" if probe_result.text else "empty"
        except Exception as e:
            result["llm"]["probe"] = f"failed: {type(e).__name__}"
            result["status"] = "degraded"

    # System metrics if psutil is available
    try:
        import psutil

        process = psutil.Process()
        result["system"] = {
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "threads": process.num_threads(),
        }
    except ImportError:
        result["system"] = {"note": "psutil not installed"}

    return result


@router.get("/health/sli", tags=["Health"], summary="Service level indicators")
async def health_sli() -> dict[str, Any]:
    """Service Level Indicator report with SLO compliance.

    Returns current SLIs compared against configured SLOs.
    Each SLI includes:
    - current: Current measured value
    - target: SLO target value
    - status: "ok" if meeting SLO, "warning", or "critical"
    """
    from hfl.metrics import get_metrics

    slo = config.slo
    sli = get_metrics().get_sli()

    # Calculate compliance for each indicator
    def check_latency(current: float, target: float) -> str:
        if current == 0.0:  # No data
            return "unknown"
        if current <= target:
            return "ok"
        if current <= target * 1.5:  # Within 150% of target
            return "warning"
        return "critical"

    def check_rate(current: float, target: float, lower_is_better: bool = True) -> str:
        if lower_is_better:
            if current <= target:
                return "ok"
            if current <= target * 2:
                return "warning"
            return "critical"
        else:
            if current >= target:
                return "ok"
            if current >= target * 0.5:
                return "warning"
            return "critical"

    # Memory usage check
    memory_usage = 0.0
    memory_status = "unknown"
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        total_memory = psutil.virtual_memory().total
        memory_usage = memory_info.rss / total_memory
        if memory_usage < slo.memory_warning_threshold:
            memory_status = "ok"
        elif memory_usage < slo.memory_critical_threshold:
            memory_status = "warning"
        else:
            memory_status = "critical"
    except ImportError:
        pass

    # Overall status: worst of all indicators
    statuses = []
    indicators = {
        "latency_p50": {
            "current_ms": sli["latency_p50_ms"],
            "target_ms": slo.latency_p50_ms,
            "status": check_latency(sli["latency_p50_ms"], slo.latency_p50_ms),
        },
        "latency_p95": {
            "current_ms": sli["latency_p95_ms"],
            "target_ms": slo.latency_p95_ms,
            "status": check_latency(sli["latency_p95_ms"], slo.latency_p95_ms),
        },
        "latency_p99": {
            "current_ms": sli["latency_p99_ms"],
            "target_ms": slo.latency_p99_ms,
            "status": check_latency(sli["latency_p99_ms"], slo.latency_p99_ms),
        },
        "error_rate": {
            "current": sli["error_rate"],
            "target": slo.error_rate_target,
            "status": check_rate(sli["error_rate"], slo.error_rate_target),
        },
        "availability": {
            "current": sli["availability"],
            "target": slo.availability_target,
            "status": check_rate(
                sli["availability"],
                slo.availability_target,
                lower_is_better=False,
            ),
        },
        "memory": {
            "current": memory_usage,
            "warning_threshold": slo.memory_warning_threshold,
            "critical_threshold": slo.memory_critical_threshold,
            "status": memory_status,
        },
    }

    # Collect all statuses for overall determination
    for ind in indicators.values():
        statuses.append(ind["status"])

    if "critical" in statuses:
        overall_status = "critical"
    elif "warning" in statuses:
        overall_status = "warning"
    elif "unknown" in statuses and all(s in ("ok", "unknown") for s in statuses):
        overall_status = "ok"  # Unknown with no failures is ok
    else:
        overall_status = "ok"

    return {
        "status": overall_status,
        "slo_version": "1.0",
        "indicators": indicators,
        "summary": {
            "total_requests": sli["total_requests"],
            "error_count": sli["error_count"],
            "sample_count": sli["sample_count"],
            "uptime_seconds": sli["uptime_seconds"],
            "throughput_rps": sli["throughput_rps"],
        },
    }
