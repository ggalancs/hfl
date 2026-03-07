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
