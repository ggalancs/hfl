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
from collections import defaultdict, deque
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
    requests_by_endpoint: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    requests_by_status: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    requests_by_method: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tokens_generated: int = 0
    tokens_input: int = 0
    model_loads: int = 0
    model_unloads: int = 0
    errors_by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Histograms using deque for O(1) FIFO eviction (circular buffer)
    _generation_latencies_ms: deque[float] = field(
        default_factory=lambda: deque(maxlen=1000), repr=False
    )
    _model_load_latencies_ms: deque[float] = field(
        default_factory=lambda: deque(maxlen=1000), repr=False
    )
    _request_latencies_ms: deque[float] = field(
        default_factory=lambda: deque(maxlen=1000), repr=False
    )

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

            # deque handles FIFO eviction automatically via maxlen
            self._request_latencies_ms.append(duration_ms)

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

            # deque handles FIFO eviction automatically via maxlen
            self._generation_latencies_ms.append(duration_ms)

    def record_model_load(self, model_name: str, duration_ms: float) -> None:
        """Record a model load.

        Args:
            model_name: Name of the loaded model
            duration_ms: Load duration in milliseconds
        """
        with self._lock:
            self.model_loads += 1

            # deque handles FIFO eviction automatically via maxlen
            self._model_load_latencies_ms.append(duration_ms)

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

    def _percentile(self, values: deque[float] | list[float], p: float) -> float:
        """Calculate percentile using linear interpolation.

        Uses the standard linear interpolation method for accurate percentiles.

        Args:
            values: Collection of values
            p: Percentile as decimal (0.0 to 1.0)

        Returns:
            Interpolated percentile value
        """
        if not values:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n == 1:
            return sorted_values[0]

        # Calculate the index with linear interpolation
        k = (n - 1) * p
        f = int(k)  # Floor
        c = f + 1 if f < n - 1 else f  # Ceiling (capped)

        if f == c:
            return sorted_values[f]

        # Linear interpolation between floor and ceiling values
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

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

            # Inference dispatcher concurrency. Pulled from the shared
            # dispatcher snapshot so /metrics reflects the same state
            # /healthz uses. Read-only — exporting these every scrape
            # is cheap because ``DispatcherSnapshot`` is just six ints.
            try:
                from hfl.core import get_dispatcher

                snap = get_dispatcher().snapshot()
                lines.append("")
                lines.append(
                    "# HELP hfl_inference_concurrency_max "
                    "Configured max in-flight inference requests"
                )
                lines.append("# TYPE hfl_inference_concurrency_max gauge")
                lines.append(f"hfl_inference_concurrency_max {snap.max_inflight}")
                lines.append("")
                lines.append(
                    "# HELP hfl_inference_concurrency_inflight "
                    "Inference requests currently executing"
                )
                lines.append("# TYPE hfl_inference_concurrency_inflight gauge")
                lines.append(f"hfl_inference_concurrency_inflight {snap.in_flight}")
                lines.append("")
                lines.append(
                    "# HELP hfl_inference_queue_depth Inference requests waiting for a slot"
                )
                lines.append("# TYPE hfl_inference_queue_depth gauge")
                lines.append(f"hfl_inference_queue_depth {snap.depth}")
            except Exception:  # pragma: no cover — config not loaded in some tests
                pass

            # Per-endpoint requests
            if self.requests_by_endpoint:
                lines.append("")
                lines.append("# HELP hfl_requests_by_endpoint_total Requests by endpoint")
                lines.append("# TYPE hfl_requests_by_endpoint_total counter")
                for endpoint, count in self.requests_by_endpoint.items():
                    lines.append(f'hfl_requests_by_endpoint_total{{endpoint="{endpoint}"}} {count}')

            # Per-status requests
            if self.requests_by_status:
                lines.append("")
                lines.append("# HELP hfl_requests_by_status_total Requests by status")
                lines.append("# TYPE hfl_requests_by_status_total counter")
                for status, count in self.requests_by_status.items():
                    lines.append(f'hfl_requests_by_status_total{{status="{status}"}} {count}')

            # Errors
            if self.errors_by_type:
                lines.append("")
                lines.append("# HELP hfl_errors_total Errors by type")
                lines.append("# TYPE hfl_errors_total counter")
                for error_type, count in self.errors_by_type.items():
                    lines.append(f'hfl_errors_total{{type="{error_type}"}} {count}')

            # Latency summaries
            if self._request_latencies_ms:
                lines.append("")
                lines.append("# HELP hfl_request_latency_ms Request latency in milliseconds")
                lines.append("# TYPE hfl_request_latency_ms summary")
                p50 = self._percentile(self._request_latencies_ms, 0.5)
                p95 = self._percentile(self._request_latencies_ms, 0.95)
                p99 = self._percentile(self._request_latencies_ms, 0.99)
                lines.append(f'hfl_request_latency_ms{{quantile="0.5"}} {p50:.2f}')
                lines.append(f'hfl_request_latency_ms{{quantile="0.95"}} {p95:.2f}')
                lines.append(f'hfl_request_latency_ms{{quantile="0.99"}} {p99:.2f}')

            if self._generation_latencies_ms:
                lines.append("")
                lines.append("# HELP hfl_generation_latency_ms Generation latency in milliseconds")
                lines.append("# TYPE hfl_generation_latency_ms summary")
                p50 = self._percentile(self._generation_latencies_ms, 0.5)
                p95 = self._percentile(self._generation_latencies_ms, 0.95)
                p99 = self._percentile(self._generation_latencies_ms, 0.99)
                lines.append(f'hfl_generation_latency_ms{{quantile="0.5"}} {p50:.2f}')
                lines.append(f'hfl_generation_latency_ms{{quantile="0.95"}} {p95:.2f}')
                lines.append(f'hfl_generation_latency_ms{{quantile="0.99"}} {p99:.2f}')

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
                    "request_p95_ms": self._percentile(self._request_latencies_ms, 0.95),
                    "request_p99_ms": self._percentile(self._request_latencies_ms, 0.99),
                    "generation_p50_ms": self._percentile(self._generation_latencies_ms, 0.5),
                    "generation_p95_ms": self._percentile(self._generation_latencies_ms, 0.95),
                },
            }

    def get_sli(self) -> dict[str, Any]:
        """Calculate Service Level Indicators from current metrics.

        Returns:
            Dictionary with SLI values:
            - latency_p50_ms: 50th percentile request latency
            - latency_p95_ms: 95th percentile request latency
            - latency_p99_ms: 99th percentile request latency
            - error_rate: Ratio of 5xx responses to total requests
            - availability: Estimated availability based on error rate
            - throughput_rps: Requests per second since start
            - sample_count: Number of latency samples
        """
        with self._lock:
            total_requests = self.requests_total

            # Calculate error rate (5xx responses / total)
            error_count = sum(
                count for status, count in self.requests_by_status.items() if status >= 500
            )
            error_rate = error_count / total_requests if total_requests > 0 else 0.0

            # Calculate throughput
            uptime = time.time() - self._start_time
            throughput_rps = total_requests / uptime if uptime > 0 else 0.0

            # Latency percentiles
            latency_p50 = self._percentile(self._request_latencies_ms, 0.5)
            latency_p95 = self._percentile(self._request_latencies_ms, 0.95)
            latency_p99 = self._percentile(self._request_latencies_ms, 0.99)

            # Availability = 1 - error_rate (simplified)
            availability = 1.0 - error_rate

            return {
                "latency_p50_ms": latency_p50,
                "latency_p95_ms": latency_p95,
                "latency_p99_ms": latency_p99,
                "error_rate": error_rate,
                "availability": availability,
                "throughput_rps": throughput_rps,
                "total_requests": total_requests,
                "error_count": error_count,
                "sample_count": len(self._request_latencies_ms),
                "uptime_seconds": uptime,
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


# Singleton access delegated to container for unified management


def get_metrics() -> Metrics:
    """Get the singleton Metrics instance.

    Returns:
        Metrics instance
    """
    from hfl.core.container import get_metrics as _get_metrics

    return _get_metrics()


def reset_metrics() -> None:
    """Reset the metrics singleton (for testing)."""
    from hfl.core.container import get_container

    get_container().metrics.reset()
