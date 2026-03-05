# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Metrics collection for HFL.

Provides in-memory metrics storage with Prometheus-compatible export.
Thread-safe for concurrent access.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Metrics:
    """Thread-safe in-memory metrics store.

    Tracks request counts, latencies, token counts, and errors.
    Supports Prometheus and JSON export formats.
    """

    # Lock for thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Start time for uptime calculation
    _start_time: float = field(default_factory=time.time, repr=False)

    # Counters
    requests_total: int = 0
    requests_by_endpoint: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    requests_by_status: dict[int, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    requests_by_method: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    tokens_generated: int = 0
    tokens_input: int = 0
    model_loads: int = 0
    model_unloads: int = 0
    errors_by_type: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    # Histograms (simplified - store recent values)
    _generation_latencies_ms: list[float] = field(default_factory=list, repr=False)
    _model_load_latencies_ms: list[float] = field(default_factory=list, repr=False)
    _request_latencies_ms: list[float] = field(default_factory=list, repr=False)
    _max_histogram_size: int = 1000

    def record_request(
        self,
        endpoint: str,
        method: str,
        status: int,
        duration_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        """Record an API request.

        Args:
            endpoint: API endpoint path
            method: HTTP method
            status: HTTP status code
            duration_ms: Request duration in milliseconds
            tokens_in: Number of input tokens
            tokens_out: Number of output tokens
        """
        with self._lock:
            self.requests_total += 1
            self.requests_by_endpoint[endpoint] += 1
            self.requests_by_status[status] += 1
            self.requests_by_method[method] += 1
            self.tokens_input += tokens_in
            self.tokens_generated += tokens_out

            self._request_latencies_ms.append(duration_ms)
            if len(self._request_latencies_ms) > self._max_histogram_size:
                self._request_latencies_ms = self._request_latencies_ms[
                    -self._max_histogram_size :
                ]

    def record_generation(
        self,
        duration_ms: float,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        """Record a text generation.

        Args:
            duration_ms: Generation duration in milliseconds
            tokens_in: Number of input tokens
            tokens_out: Number of output tokens
        """
        with self._lock:
            self.tokens_input += tokens_in
            self.tokens_generated += tokens_out

            self._generation_latencies_ms.append(duration_ms)
            if len(self._generation_latencies_ms) > self._max_histogram_size:
                self._generation_latencies_ms = self._generation_latencies_ms[
                    -self._max_histogram_size :
                ]

    def record_model_load(self, model_name: str, duration_ms: float) -> None:
        """Record a model load.

        Args:
            model_name: Name of the loaded model
            duration_ms: Load duration in milliseconds
        """
        with self._lock:
            self.model_loads += 1

            self._model_load_latencies_ms.append(duration_ms)
            if len(self._model_load_latencies_ms) > self._max_histogram_size:
                self._model_load_latencies_ms = self._model_load_latencies_ms[
                    -self._max_histogram_size :
                ]

    def record_model_unload(self) -> None:
        """Record a model unload."""
        with self._lock:
            self.model_unloads += 1

    def record_error(self, error_type: str) -> None:
        """Record an error.

        Args:
            error_type: Type/class of the error
        """
        with self._lock:
            self.errors_by_type[error_type] += 1

    def _percentile(self, values: list[float], p: float) -> float:
        """Calculate percentile of a sorted list."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * p)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        with self._lock:
            lines = []

            # Uptime
            uptime = time.time() - self._start_time
            lines.append("# HELP hfl_uptime_seconds Time since server start")
            lines.append("# TYPE hfl_uptime_seconds gauge")
            lines.append(f"hfl_uptime_seconds {uptime:.2f}")

            # Request counters
            lines.append("")
            lines.append("# HELP hfl_requests_total Total number of requests")
            lines.append("# TYPE hfl_requests_total counter")
            lines.append(f"hfl_requests_total {self.requests_total}")

            # Token counters
            lines.append("")
            lines.append("# HELP hfl_tokens_generated_total Total tokens generated")
            lines.append("# TYPE hfl_tokens_generated_total counter")
            lines.append(f"hfl_tokens_generated_total {self.tokens_generated}")

            lines.append("")
            lines.append("# HELP hfl_tokens_input_total Total input tokens")
            lines.append("# TYPE hfl_tokens_input_total counter")
            lines.append(f"hfl_tokens_input_total {self.tokens_input}")

            # Model loads
            lines.append("")
            lines.append("# HELP hfl_model_loads_total Total model loads")
            lines.append("# TYPE hfl_model_loads_total counter")
            lines.append(f"hfl_model_loads_total {self.model_loads}")

            # Per-endpoint requests
            if self.requests_by_endpoint:
                lines.append("")
                lines.append(
                    "# HELP hfl_requests_by_endpoint_total Requests by endpoint"
                )
                lines.append("# TYPE hfl_requests_by_endpoint_total counter")
                for endpoint, count in self.requests_by_endpoint.items():
                    lines.append(
                        f'hfl_requests_by_endpoint_total{{endpoint="{endpoint}"}} {count}'
                    )

            # Per-status requests
            if self.requests_by_status:
                lines.append("")
                lines.append("# HELP hfl_requests_by_status_total Requests by status")
                lines.append("# TYPE hfl_requests_by_status_total counter")
                for status, count in self.requests_by_status.items():
                    lines.append(
                        f'hfl_requests_by_status_total{{status="{status}"}} {count}'
                    )

            # Errors
            if self.errors_by_type:
                lines.append("")
                lines.append("# HELP hfl_errors_total Errors by type")
                lines.append("# TYPE hfl_errors_total counter")
                for error_type, count in self.errors_by_type.items():
                    lines.append(
                        f'hfl_errors_total{{type="{error_type}"}} {count}'
                    )

            # Latency summaries
            if self._request_latencies_ms:
                lines.append("")
                lines.append(
                    "# HELP hfl_request_latency_ms Request latency in milliseconds"
                )
                lines.append("# TYPE hfl_request_latency_ms summary")
                lines.append(
                    f'hfl_request_latency_ms{{quantile="0.5"}} {self._percentile(self._request_latencies_ms, 0.5):.2f}'
                )
                lines.append(
                    f'hfl_request_latency_ms{{quantile="0.95"}} {self._percentile(self._request_latencies_ms, 0.95):.2f}'
                )
                lines.append(
                    f'hfl_request_latency_ms{{quantile="0.99"}} {self._percentile(self._request_latencies_ms, 0.99):.2f}'
                )

            if self._generation_latencies_ms:
                lines.append("")
                lines.append(
                    "# HELP hfl_generation_latency_ms Generation latency in milliseconds"
                )
                lines.append("# TYPE hfl_generation_latency_ms summary")
                lines.append(
                    f'hfl_generation_latency_ms{{quantile="0.5"}} {self._percentile(self._generation_latencies_ms, 0.5):.2f}'
                )
                lines.append(
                    f'hfl_generation_latency_ms{{quantile="0.95"}} {self._percentile(self._generation_latencies_ms, 0.95):.2f}'
                )
                lines.append(
                    f'hfl_generation_latency_ms{{quantile="0.99"}} {self._percentile(self._generation_latencies_ms, 0.99):.2f}'
                )

            return "\n".join(lines)

    def export_json(self) -> dict[str, Any]:
        """Export metrics as JSON.

        Returns:
            Dictionary with metrics data
        """
        with self._lock:
            return {
                "uptime_seconds": time.time() - self._start_time,
                "requests_total": self.requests_total,
                "tokens_generated": self.tokens_generated,
                "tokens_input": self.tokens_input,
                "model_loads": self.model_loads,
                "model_unloads": self.model_unloads,
                "requests_by_endpoint": dict(self.requests_by_endpoint),
                "requests_by_status": dict(self.requests_by_status),
                "requests_by_method": dict(self.requests_by_method),
                "errors_by_type": dict(self.errors_by_type),
                "latencies": {
                    "request_p50_ms": self._percentile(self._request_latencies_ms, 0.5),
                    "request_p95_ms": self._percentile(
                        self._request_latencies_ms, 0.95
                    ),
                    "request_p99_ms": self._percentile(
                        self._request_latencies_ms, 0.99
                    ),
                    "generation_p50_ms": self._percentile(
                        self._generation_latencies_ms, 0.5
                    ),
                    "generation_p95_ms": self._percentile(
                        self._generation_latencies_ms, 0.95
                    ),
                },
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.requests_total = 0
            self.requests_by_endpoint.clear()
            self.requests_by_status.clear()
            self.requests_by_method.clear()
            self.tokens_generated = 0
            self.tokens_input = 0
            self.model_loads = 0
            self.model_unloads = 0
            self.errors_by_type.clear()
            self._generation_latencies_ms.clear()
            self._model_load_latencies_ms.clear()
            self._request_latencies_ms.clear()
            self._start_time = time.time()


# Singleton instance
_metrics: Metrics | None = None


def get_metrics() -> Metrics:
    """Get the singleton Metrics instance.

    Returns:
        Metrics instance
    """
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
    return _metrics


def reset_metrics() -> None:
    """Reset the metrics singleton (for testing)."""
    global _metrics
    _metrics = None
