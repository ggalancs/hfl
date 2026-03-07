# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Core infrastructure for HFL."""

from hfl.core.container import (
    Container,
    Singleton,
    get_config,
    get_container,
    get_event_bus,
    get_metrics,
    get_registry,
    get_state,
    reset_container,
)
from hfl.core.tracing import (
    RequestContext,
    clear_request_id,
    format_log_prefix,
    generate_request_id,
    get_request_id,
    get_trace_context,
    set_request_id,
    set_trace_context,
    with_request_id,
)

__all__ = [
    # Container
    "Container",
    "Singleton",
    "get_config",
    "get_container",
    "get_event_bus",
    "get_metrics",
    "get_registry",
    "get_state",
    "reset_container",
    # Tracing
    "RequestContext",
    "clear_request_id",
    "format_log_prefix",
    "generate_request_id",
    "get_request_id",
    "get_trace_context",
    "set_request_id",
    "set_trace_context",
    "with_request_id",
]
