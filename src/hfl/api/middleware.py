# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Middleware for logging, rate limiting, and error handling.

PRIVACY (R6 - Legal Audit):
This middleware implements privacy-safe logging that:
- NEVER logs request content (prompts)
- NEVER logs response content (AI outputs)
- NEVER logs authentication headers
- Only logs metadata: method, path, status, duration
"""

import ipaddress
import time
from collections import defaultdict
from typing import Any, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from hfl.logging_config import get_logger, log_request, set_request_id
from hfl.metrics import get_metrics

logger = get_logger()

# Trusted proxy networks (RFC 1918 private ranges + localhost)
# Only trust X-Forwarded-For from these networks
TRUSTED_PROXY_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),  # Localhost
    ipaddress.ip_network("10.0.0.0/8"),  # Private Class A
    ipaddress.ip_network("172.16.0.0/12"),  # Private Class B
    ipaddress.ip_network("192.168.0.0/16"),  # Private Class C
    ipaddress.ip_network("::1/128"),  # IPv6 localhost
    ipaddress.ip_network("fc00::/7"),  # IPv6 private
]


def _is_trusted_proxy(ip: str) -> bool:
    """Check if IP is from a trusted proxy network."""
    try:
        addr = ipaddress.ip_address(ip)
        return any(addr in network for network in TRUSTED_PROXY_NETWORKS)
    except ValueError:
        return False


class RequestLogger(BaseHTTPMiddleware):
    """
    Privacy-safe logging middleware with request tracing.

    IMPORTANT: This logger is designed to NEVER log:
    - Request bodies (contain user prompts)
    - Response bodies (contain model outputs)
    - Authorization headers (contain tokens)
    - User-Agent or other identifying headers

    Only basic metadata is logged for debugging and metrics.
    Features:
    - Request ID tracing (X-Request-ID header)
    - Structured logging compatible
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        # Set request ID from header or generate new one
        incoming_id = request.headers.get("X-Request-ID")
        request_id = set_request_id(incoming_id)

        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        # Add request ID to response headers for tracing
        response.headers["X-Request-ID"] = request_id

        # Observability headers for the inference dispatcher (spec §5.3).
        # Exposed on every response (including health/metrics paths) so
        # monitoring dashboards and agent clients can trivially observe
        # live backpressure state.
        try:
            from hfl.core import get_dispatcher

            snap = get_dispatcher().snapshot()
            response.headers["X-Queue-Depth"] = str(snap.depth)
            response.headers["X-Queue-In-Flight"] = str(snap.in_flight)
            response.headers["X-Queue-Max-Inflight"] = str(snap.max_inflight)
            response.headers["X-Queue-Max-Size"] = str(snap.max_queued)
        except Exception:  # pragma: no cover — defensive
            pass

        # Log request with structured data (privacy-safe)
        # R6 - Privacy compliance: no personal data in logs
        log_request(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
        )

        # Record metrics for monitoring
        get_metrics().record_request(
            endpoint=request.url.path,
            method=request.method,
            status=response.status_code,
            duration_ms=duration_ms,
        )

        return response


# Global reference for testing - set when middleware is created
_rate_limiter_instance: "RateLimitMiddleware | None" = None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware using sliding window.

    Tracks requests per client IP and rejects requests that exceed
    the configured limit within the time window.

    Health check endpoints (/health/*) are excluded from rate limiting
    to ensure load balancers and orchestrators can always check status.
    """

    # Paths excluded from rate limiting (used by health checks, monitoring)
    EXCLUDED_PREFIXES = ("/health",)

    def __init__(
        self,
        app: Any,
        requests_per_window: int = 60,
        window_seconds: int = 60,
    ) -> None:
        super().__init__(app)
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self._request_counts: dict[str, list[float]] = defaultdict(list)
        self._request_counter = 0
        self._cleanup_interval = 1000  # Clean up stale IPs every N requests
        # Store global reference for testing
        global _rate_limiter_instance
        _rate_limiter_instance = self

    def _is_excluded(self, path: str) -> bool:
        """Check if path is excluded from rate limiting."""
        return path.startswith(self.EXCLUDED_PREFIXES)

    def reset(self) -> None:
        """Clear all rate limit tracking. Used for testing."""
        self._request_counts.clear()

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP from request headers or connection.

        Only trusts X-Forwarded-For header if the request comes from
        a trusted proxy network (private IP ranges). This prevents
        attackers from spoofing their IP via the header.

        For trusted proxies, takes the rightmost untrusted IP from
        the X-Forwarded-For chain (the actual client).

        Invalid IP addresses in X-Forwarded-For are logged and skipped.
        """
        # Get direct connection IP
        connection_ip = request.client.host if request.client else "unknown"

        # Only trust X-Forwarded-For if request came from trusted proxy
        if connection_ip != "unknown" and _is_trusted_proxy(connection_ip):
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                # Parse all IPs in the chain
                for ip_str in reversed(forwarded.split(",")):
                    ip_str = ip_str.strip()
                    if not ip_str:
                        continue

                    # Validate IP format before using
                    try:
                        ipaddress.ip_address(ip_str)
                    except ValueError:
                        logger.warning("Invalid IP in X-Forwarded-For: %s", ip_str)
                        continue

                    # Find rightmost IP that is NOT a trusted proxy
                    if not _is_trusted_proxy(ip_str):
                        return ip_str

                # If all IPs are trusted, use the first valid one
                for ip_str in forwarded.split(","):
                    ip_str = ip_str.strip()
                    try:
                        ipaddress.ip_address(ip_str)
                        return ip_str
                    except ValueError:
                        continue

        return connection_ip

    def _is_rate_limited(self, client_ip: str) -> tuple[bool, int]:
        """Check if client is rate limited.

        Returns:
            Tuple of (is_limited, remaining_requests)
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Remove old timestamps outside the window
        self._request_counts[client_ip] = [
            ts for ts in self._request_counts[client_ip] if ts > window_start
        ]

        # Check if over limit
        current_count = len(self._request_counts[client_ip])
        remaining = max(0, self.requests_per_window - current_count)

        if current_count >= self.requests_per_window:
            return True, 0

        # Record this request
        self._request_counts[client_ip].append(now)

        # Periodic cleanup of stale IPs
        self._request_counter += 1
        if self._request_counter >= self._cleanup_interval:
            self._request_counter = 0
            stale_ips = [
                ip
                for ip, timestamps in self._request_counts.items()
                if not timestamps or timestamps[-1] < window_start
            ]
            for ip in stale_ips:
                del self._request_counts[ip]

        return False, remaining - 1

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        # Skip rate limiting for excluded paths (health checks, etc.)
        if self._is_excluded(request.url.path):
            response: Response = await call_next(request)
            return response

        client_ip = self._get_client_ip(request)
        is_limited, remaining = self._is_rate_limited(client_ip)

        reset_at = int(time.time()) + self.window_seconds

        if is_limited:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "error": "Rate limit exceeded",
                        "code": "RATE_LIMIT_EXCEEDED",
                        "category": "rate_limit",
                        "retryable": True,
                        "details": {
                            "retry_after_seconds": self.window_seconds,
                            "window_seconds": self.window_seconds,
                        },
                    }
                },
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(self.requests_per_window),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at),
                    "X-RateLimit-Window": str(self.window_seconds),
                },
            )

        response = await call_next(request)

        # Add rate limit headers on every response so the client can plan
        # back-off proportionally (spec §5.2).
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_at)
        response.headers["X-RateLimit-Window"] = str(self.window_seconds)

        return response


def reset_rate_limiter() -> None:
    """Reset rate limiter storage. Used for testing."""
    global _rate_limiter_instance
    if _rate_limiter_instance is not None:
        _rate_limiter_instance.reset()
