# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""OpenTelemetry tracing (Phase 17 P2 — V2 row 24).

Thin wrapper around ``opentelemetry`` so the rest of HFL can write

    with trace_span("engine.generate", attributes={"model": name}):
        ...

without caring whether OTEL is configured. When it isn't (default)
the helper returns a no-op context manager so the call site stays
zero-cost.

Configuration:

- ``HFL_OTEL_ENABLED=true`` turns tracing on.
- ``HFL_OTEL_EXPORTER_ENDPOINT`` points at an OTLP/HTTP collector
  (default ``http://localhost:4318``).
- ``HFL_OTEL_SERVICE_NAME`` stamps the service attribute on every
  span (default ``hfl``).

The ``[otel]`` extra pulls ``opentelemetry-api`` + ``-sdk`` +
``-exporter-otlp-proto-http``.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

logger = logging.getLogger(__name__)

__all__ = [
    "configure_tracing",
    "reset_tracing",
    "trace_span",
    "is_enabled",
]


_tracer: Any = None
_configured: bool = False


def is_enabled() -> bool:
    """Return True when the OTEL SDK is loaded and a tracer exists."""
    return _tracer is not None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def configure_tracing(
    *,
    enabled: bool | None = None,
    endpoint: str | None = None,
    service_name: str | None = None,
) -> bool:
    """Install the OTEL tracer provider + OTLP/HTTP exporter.

    Returns True when tracing is active after the call. Idempotent —
    calling twice replaces the tracer with a fresh one pointing at
    whatever configuration the second call carried.
    """
    global _tracer, _configured

    want = enabled if enabled is not None else _env_flag("HFL_OTEL_ENABLED")
    if not want:
        _tracer = None
        _configured = True
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "HFL_OTEL_ENABLED is set but opentelemetry SDK is not installed. "
            "Install with `pip install 'hfl[otel]'`."
        )
        _tracer = None
        _configured = True
        return False

    endpoint = endpoint or os.environ.get(
        "HFL_OTEL_EXPORTER_ENDPOINT", "http://localhost:4318/v1/traces"
    )
    service_name = service_name or os.environ.get("HFL_OTEL_SERVICE_NAME", "hfl")

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("hfl")
    _configured = True
    logger.info("OpenTelemetry tracing active → %s (service=%s)", endpoint, service_name)
    return True


def reset_tracing() -> None:
    """Test hook — drops the tracer so each test starts clean."""
    global _tracer, _configured
    _tracer = None
    _configured = False


@contextmanager
def trace_span(name: str, *, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """Produce a span when tracing is enabled, a no-op otherwise.

    Callers don't need to check ``is_enabled`` — the context
    manager yields ``None`` in the disabled case.
    """
    if not _configured:
        configure_tracing()
    if _tracer is None:
        with nullcontext() as noop:
            yield noop
        return
    with _tracer.start_as_current_span(name, attributes=attributes) as span:
        yield span
