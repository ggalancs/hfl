# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Global exception handlers for the FastAPI app.

Maps domain exceptions (HFLError hierarchy) to HTTP responses,
so routes can raise domain exceptions instead of HTTPException.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from hfl.api.errors import render_envelope
from hfl.exceptions import HFLError
from hfl.logging_config import get_request_id

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

        # API-3: shape the body for the OpenAI/Anthropic dialect on /v1/*
        # routes; /api/* and everything else keep the flat HFL body unchanged.
        body = render_envelope(request.url.path, status_code, body)
        return JSONResponse(status_code=status_code, content=body)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """API-4: emit a per-dialect 400 for an invalid request body on the
        OpenAI/Anthropic surfaces — real OpenAI/Anthropic return 400 (not
        FastAPI's default 422 ``{"detail": [...]}``). ``/api/*`` and everything
        else keep the native 422 shape.
        """
        path = request.url.path
        if not path.startswith("/v1/"):
            return await request_validation_exception_handler(request, exc)

        errors = jsonable_encoder(exc.errors())
        first = errors[0] if errors else {}
        loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        msg = first.get("msg") or "Invalid request"
        message = f"{loc}: {msg}" if loc else msg
        flat = {
            "error": message,
            "code": "VALIDATION_ERROR",
            "category": "validation",
            "retryable": False,
            "details": {"errors": errors},
            "request_id": get_request_id(),
        }
        return JSONResponse(status_code=400, content=render_envelope(path, 400, flat))

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
