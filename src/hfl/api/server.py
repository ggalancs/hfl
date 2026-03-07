# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
REST API server compatible with OpenAI and Ollama.

Implemented endpoints:
  OpenAI:
    POST /v1/chat/completions
    POST /v1/completions
    GET  /v1/models

  Ollama-native:
    POST /api/generate
    POST /api/chat
    GET  /api/tags
    POST /api/pull
    DELETE /api/delete

Legal Compliance (R9 - Audit):
- Disclaimer header in all AI responses

Security:
- Optional API key authentication via --api-key flag
"""

import secrets
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Callable

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from hfl.api.routes_health import router as health_router
from hfl.api.routes_metrics import router as metrics_router
from hfl.api.routes_native import router as native_router
from hfl.api.routes_openai import router as openai_router
from hfl.api.routes_tts import router as tts_router
from hfl.api.state import get_state
from hfl.config import config


# API Key Authentication Middleware
class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware that validates API key if configured."""

    # Endpoints that don't require authentication (exact match)
    PUBLIC_ENDPOINTS = {"/", "/api/version"}

    # Path prefixes that don't require authentication
    PUBLIC_PREFIXES = ("/health", "/metrics")

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        state = get_state()

        # Skip auth if no API key is configured
        if not state.api_key:
            response: Response = await call_next(request)
            return response

        # Skip auth for public endpoints (exact match)
        if request.url.path in self.PUBLIC_ENDPOINTS:
            response = await call_next(request)
            return response

        # Skip auth for public prefixes (e.g., /health, /health/ready, /health/live)
        if request.url.path.startswith(self.PUBLIC_PREFIXES):
            response = await call_next(request)
            return response

        # Check for API key in Authorization header (Bearer token)
        # Use constant-time comparison to prevent timing attacks
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if secrets.compare_digest(token.encode(), state.api_key.encode()):
                response = await call_next(request)
                return response

        # Check for API key in X-API-Key header
        api_key_header = request.headers.get("X-API-Key", "")
        if api_key_header and secrets.compare_digest(api_key_header.encode(), state.api_key.encode()):
            response = await call_next(request)
            return response

        # Authentication failed
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid or missing API key"},
            headers={"WWW-Authenticate": "Bearer"},
        )


# R9 - Disclaimer Middleware (Legal Audit)
# Adds disclaimer header to all AI responses
class DisclaimerMiddleware(BaseHTTPMiddleware):
    """Middleware that adds disclaimer to AI endpoint responses."""

    AI_ENDPOINTS = {
        # LLM endpoints
        "/v1/chat/completions",
        "/v1/completions",
        "/api/generate",
        "/api/chat",
        # TTS endpoints
        "/v1/audio/speech",
        "/api/tts",
    }

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        response: Response = await call_next(request)
        # Only add disclaimer to generation endpoints
        if request.url.path in self.AI_ENDPOINTS:
            response.headers["X-AI-Disclaimer"] = (
                "AI-generated content. May be inaccurate or inappropriate. "
                "User assumes all responsibility for use of outputs."
            )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Server lifecycle."""
    yield
    # Cleanup on shutdown
    await get_state().cleanup()


app = FastAPI(
    title="hfl API",
    description="OpenAI and Ollama compatible API for HuggingFace models",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS - Configurable via config.py
# Use ["*"] if cors_allow_all is True, otherwise use explicit origins
_cors_origins = ["*"] if config.cors_allow_all else config.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=config.cors_allow_credentials,
    allow_methods=config.cors_allow_methods,
    allow_headers=config.cors_allow_headers,
)

# R9 - Add disclaimer middleware
app.add_middleware(DisclaimerMiddleware)

# Add API key authentication middleware
app.add_middleware(APIKeyMiddleware)

# Optional rate limiting (disabled by default)
if config.rate_limit_enabled:
    from hfl.api.middleware import RateLimitMiddleware

    app.add_middleware(
        RateLimitMiddleware,
        requests_per_window=config.rate_limit_requests,
        window_seconds=config.rate_limit_window,
    )

# Request logging and metrics recording
from hfl.api.middleware import RequestLogger

app.add_middleware(RequestLogger)

app.include_router(openai_router)
app.include_router(native_router)
app.include_router(tts_router)
app.include_router(health_router)
app.include_router(metrics_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "hfl is running"}


def start_server(
    host: str | None = None,
    port: int | None = None,
    api_key: str | None = None,
) -> None:
    """Start the API server.

    Args:
        host: Host address to bind (default: from config)
        port: Port number (default: from config)
        api_key: Optional API key for authentication. If set, all requests
                 must include either:
                 - Authorization: Bearer <api_key>
                 - X-API-Key: <api_key>
    """
    get_state().api_key = api_key
    uvicorn.run(
        app,
        host=host or config.host,
        port=port or config.port,
        log_level="info",
    )
