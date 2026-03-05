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

import time
from collections import defaultdict
from typing import Any, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from hfl.logging_config import get_logger, log_request, set_request_id

logger = get_logger()


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

        # Log request with structured data (privacy-safe)
        # R6 - Privacy compliance: no personal data in logs
        log_request(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
        )

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware using sliding window.

    Tracks requests per client IP and rejects requests that exceed
    the configured limit within the time window.
    """

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

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP from request headers or connection.

        Respects X-Forwarded-For for proxy scenarios but falls back
        to connection IP for direct connections.
        """
        # Check X-Forwarded-For header (first IP in the chain)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        # Fall back to connection IP
        if request.client:
            return request.client.host
        return "unknown"

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
        return False, remaining - 1

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        client_ip = self._get_client_ip(request)
        is_limited, remaining = self._is_rate_limited(client_ip)

        if is_limited:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": f"Rate limit exceeded. Try again in {self.window_seconds} seconds.",
                },
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(self.requests_per_window),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response: Response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response
