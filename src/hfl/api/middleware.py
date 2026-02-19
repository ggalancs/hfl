# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Middleware for logging, CORS and error handling.

PRIVACY (R6 - Legal Audit):
This middleware implements privacy-safe logging that:
- NEVER logs request content (prompts)
- NEVER logs response content (AI outputs)
- NEVER logs authentication headers
- Only logs metadata: method, path, status, duration
"""

import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("hfl")


class RequestLogger(BaseHTTPMiddleware):
    """
    Privacy-safe logging middleware.

    IMPORTANT: This logger is designed to NEVER log:
    - Request bodies (contain user prompts)
    - Response bodies (contain model outputs)
    - Authorization headers (contain tokens)
    - User-Agent or other identifying headers

    Only basic metadata is logged for debugging and metrics.
    """

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        # Only log metadata, NEVER the body or sensitive headers
        # R6 - Privacy compliance: no personal data in logs
        logger.info(
            "method=%s path=%s status=%d duration=%.3fs",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            # DO NOT include: request body, auth headers, user-agent, real IP
        )
        return response
