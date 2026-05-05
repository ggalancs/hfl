# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Global exception handlers for the FastAPI app.

Maps domain exceptions (HFLError hierarchy) to HTTP responses,
so routes can raise domain exceptions instead of HTTPException.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from hfl.exceptions import HFLError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI app."""

    @app.exception_handler(HFLError)
    async def hfl_error_handler(request: Request, exc: HFLError) -> JSONResponse:
        """Convert HFLError subclasses to structured HTTP responses."""
        status_code = getattr(exc, "status_code", 500)

        body: dict = {
            "error": exc.message,
            "code": type(exc).__name__,
        }
        if exc.details:
            body["details"] = exc.details

        if status_code >= 500:
            logger.error("Unhandled HFLError: %s", exc)

        return JSONResponse(status_code=status_code, content=body)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all for non-HFLError exceptions that escape route handlers.

        Without this, Starlette falls back to a plain-text
        ``"Internal Server Error"`` body with no diagnostic info — the
        caller can't distinguish an engine crash from a wiring bug, and
        the server log carries only the traceback at whatever level
        uvicorn picked. This handler:

        - Logs the full traceback via ``logger.exception`` so operators
          can correlate the 500 with a cause.
        - Returns a structured JSON envelope with the exception's class
          name and message so clients get something actionable (while
          still not leaking Python internals beyond the type name).

        This closes a long-standing diagnostic gap: a thread-pool worker
        left orphaned after a timeout can raise a ``RuntimeError`` on
        the next request, which previously surfaced as an opaque 500.
        """
        # Re-raise HTTPException so FastAPI's own handler formats it —
        # the Exception handler is registered last, so HTTPException
        # normally reaches its handler first, but be defensive in case
        # registration order ever changes.
        from fastapi import HTTPException

        if isinstance(exc, HTTPException):
            raise exc

        logger.exception(
            "unhandled exception in %s %s: %s",
            request.method,
            request.url.path,
            type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "code": "UnhandledError",
                "error_type": type(exc).__name__,
                "message": str(exc)[:500],
            },
        )
