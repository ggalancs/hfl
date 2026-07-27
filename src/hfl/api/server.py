# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
REST API server compatible with OpenAI, Ollama, and Anthropic.

Implemented endpoints:
  OpenAI:
    POST /v1/chat/completions
    POST /v1/completions
    GET  /v1/models

  Anthropic:
    POST /v1/messages

  Ollama-native:
    POST /api/generate
    POST /api/chat
    GET  /api/tags
    POST /api/pull
    (model deletion is CLI-only — `hfl rm`; there is deliberately no
     DELETE /api/delete route, so the API cannot destroy local models.)

Legal Compliance (R9 - Audit):
- Disclaimer header in all AI responses

Security:
- Optional API key authentication via --api-key flag
"""

import asyncio
import logging
import os
import secrets
from collections import OrderedDict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Callable

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from hfl import __version__
from hfl.api.exception_handlers import register_exception_handlers
from hfl.api.middleware import RequestBodyLimitMiddleware, RequestLogger
from hfl.api.routes_anthropic import router as anthropic_router
from hfl.api.routes_batch import router as batch_router
from hfl.api.routes_benchmark import router as benchmark_router
from hfl.api.routes_blobs import router as blobs_router
from hfl.api.routes_compliance import router as compliance_router
from hfl.api.routes_copy import router as copy_router
from hfl.api.routes_create import router as create_router
from hfl.api.routes_discover import router as discover_router
from hfl.api.routes_draft import router as draft_router
from hfl.api.routes_embed import router as embed_router
from hfl.api.routes_health import router as health_router
from hfl.api.routes_images import router as images_router
from hfl.api.routes_lora import router as lora_router
from hfl.api.routes_metrics import router as metrics_router
from hfl.api.routes_native import router as native_router
from hfl.api.routes_openai import router as openai_router
from hfl.api.routes_openai_responses import router as openai_responses_router
from hfl.api.routes_ps import router as ps_router
from hfl.api.routes_pull import router as pull_router
from hfl.api.routes_push import router as push_router
from hfl.api.routes_recommend import router as recommend_router
from hfl.api.routes_show import router as show_router
from hfl.api.routes_smart_pull import router as smart_pull_router
from hfl.api.routes_snapshot import router as snapshot_router
from hfl.api.routes_stop import router as stop_router
from hfl.api.routes_transcribe import router as transcribe_router
from hfl.api.routes_tts import router as tts_router
from hfl.api.routes_verify import router as verify_router
from hfl.api.routes_web import router as web_router
from hfl.api.routes_ws import router as ws_router
from hfl.api.state import get_state
from hfl.config import config

logger = logging.getLogger(__name__)


# Consecutive failed-auth counters, keyed by peer address. Bounded so a
# spoofed-source flood can't grow it without limit; the map is advisory
# (a lost entry only means one attacker gets a fresh budget).
_AUTH_FAILURES: "OrderedDict[str, int]" = OrderedDict()
_AUTH_FAILURES_MAX = 4096
_AUTH_BACKOFF_AFTER = 3  # first failures answer immediately
_AUTH_BACKOFF_CAP_S = 2.0


async def _record_auth_failure(request: Request) -> None:
    """Count a failed authentication and sleep proportionally.

    The delay starts only after a few failures so a human who mistypes a
    key is not punished, and it is capped so a flood of bad keys can't be
    turned into a self-inflicted denial of service by tying up workers.
    """
    peer = request.client.host if request.client else "unknown"
    count = _AUTH_FAILURES.get(peer, 0) + 1
    _AUTH_FAILURES[peer] = count
    _AUTH_FAILURES.move_to_end(peer)
    while len(_AUTH_FAILURES) > _AUTH_FAILURES_MAX:
        _AUTH_FAILURES.popitem(last=False)

    if count > _AUTH_BACKOFF_AFTER:
        delay = min(_AUTH_BACKOFF_CAP_S, 0.1 * (2 ** (count - _AUTH_BACKOFF_AFTER - 1)))
        await asyncio.sleep(delay)
    if count in (_AUTH_BACKOFF_AFTER + 1, 25, 100):
        logger.warning("repeated API key failures from %s (%d consecutive)", peer, count)


def _clear_auth_failures(request: Request) -> None:
    """Reset a peer's counter after a successful authentication."""
    peer = request.client.host if request.client else "unknown"
    _AUTH_FAILURES.pop(peer, None)


# API Key Authentication Middleware
class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware that validates API key if configured."""

    # Endpoints that don't require authentication — EXACT match only.
    # /api/tags is intentionally kept out of the public set so auth policy
    # is deterministic (spec §5.1): when an API key is configured, /api/tags
    # always requires it, matching /api/chat.
    #
    # SEC: this used to be a ``startswith`` prefix test over
    # ``("/health", "/metrics")``. Two problems, both verified:
    #
    #  1. uvicorn does NOT normalise ``..`` in the request path, so the
    #     middleware saw ``/health/../api/chat`` verbatim and skipped auth
    #     for it. Nothing was reachable behind it (Starlette's router does
    #     not resolve ``..`` either, so such paths 404), but any future
    #     route registered under a public prefix would have turned that
    #     into a live bypass.
    #  2. A prefix test also exempts ``/health-anything`` and
    #     ``/metricsfoo``.
    #
    # An exact-match frozenset has neither property. Adding a public path
    # is now a deliberate act, and a typo fails closed.
    PUBLIC_ENDPOINTS = frozenset(
        {
            "/",
            "/api/version",
            "/healthz",
            "/health",
            "/health/live",
            "/health/ready",
            "/health/deep",
            "/health/sli",
        }
    )

    @staticmethod
    def _metrics_is_public() -> bool:
        """Whether ``/metrics`` may be served without the API key.

        Off by default. The metrics surface exposes request and token
        volumes, per-endpoint counters and live queue depth — a usage
        side-channel that has no business being readable by anyone the
        operator deliberately locked out with an API key. Operators whose
        Prometheus cannot authenticate can opt back in.
        """
        return os.environ.get("HFL_METRICS_PUBLIC", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        state = get_state()

        # Skip auth if no API key is configured
        if not state.api_key:
            response: Response = await call_next(request)
            return response

        # Skip auth for public endpoints — exact match, no prefixes.
        path = request.url.path.rstrip("/") or "/"
        if path in self.PUBLIC_ENDPOINTS:
            response = await call_next(request)
            return response

        # /metrics is authenticated unless the operator opts out.
        if path in ("/metrics", "/metrics/json") and self._metrics_is_public():
            response = await call_next(request)
            return response

        # Check for API key in Authorization header (Bearer token)
        # Use constant-time comparison to prevent timing attacks
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if secrets.compare_digest(token.encode(), state.api_key.encode()):
                _clear_auth_failures(request)
                response = await call_next(request)
                return response

        # Check for API key in X-API-Key header
        api_key_header = request.headers.get("X-API-Key", "")
        if api_key_header and secrets.compare_digest(
            api_key_header.encode(), state.api_key.encode()
        ):
            _clear_auth_failures(request)
            response = await call_next(request)
            return response

        # Authentication failed. Record it and apply a backoff before
        # answering: the global rate limiter caps request *volume* but
        # treats a 401 like any other request, so it does nothing to slow
        # a key-guessing loop specifically. A per-peer delay that grows
        # with consecutive failures makes brute force impractical while
        # staying invisible to a client that simply mistyped its key once.
        await _record_auth_failure(request)

        # Structured envelope so clients can decide retry policy without
        # parsing prose (spec §5.4).
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "error": "Invalid or missing API key",
                    "code": "UNAUTHORIZED",
                    "category": "auth",
                    "retryable": False,
                }
            },
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
        # Anthropic endpoints
        "/v1/messages",
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


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Baseline browser-hardening headers on every response.

    HFL is a JSON API, so the blast radius is small — but not zero:
    ``/api/web_fetch`` returns content derived from third-party pages and
    ``routes_web`` serves a browser-facing surface. ``nosniff`` stops a
    browser from re-interpreting a JSON body as HTML and executing it,
    ``DENY`` keeps any response out of an attacker's iframe, and
    ``no-referrer`` prevents a URL that may carry an API key in its query
    string (the WebSocket handshake form) from leaking to third parties.
    """

    HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Cross-Origin-Resource-Policy": "same-origin",
    }

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        response: Response = await call_next(request)
        for name, value in self.HEADERS.items():
            response.headers.setdefault(name, value)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Server lifecycle."""
    yield
    # Cleanup on shutdown
    await get_state().cleanup()
    # Close HTTP clients
    from hfl.hub.client import close_async_hf_client, close_hf_client

    await close_async_hf_client()
    close_hf_client()


_openapi_tags = [
    {
        "name": "OpenAI",
        "description": ("OpenAI-compatible endpoints (chat completions, completions, models)"),
    },
    {
        "name": "Anthropic",
        "description": "Anthropic Messages API-compatible endpoints",
    },
    {"name": "Ollama", "description": "Ollama-compatible endpoints (generate, chat, tags)"},
    {"name": "TTS", "description": "Text-to-speech endpoints"},
    {"name": "Health", "description": "Health check and readiness probes"},
    {"name": "Metrics", "description": "Prometheus and JSON metrics"},
]

app = FastAPI(
    title="hfl API",
    description="OpenAI, Ollama, and Anthropic compatible API for HuggingFace models",
    version=__version__,
    lifespan=lifespan,
    openapi_tags=_openapi_tags,
)

# R9 - Add disclaimer middleware
app.add_middleware(DisclaimerMiddleware)

# Baseline browser-hardening headers on every response (see the class).
app.add_middleware(SecurityHeadersMiddleware)

# Middleware execution order (Starlette runs in reverse add order):
# CORS → RequestLogger → BodyLimit → APIKey → RateLimit → Disclaimer
# CORS is added LAST so it is the OUTERMOST middleware (API-3 fix): a browser
# CORS preflight (OPTIONS) carries no Authorization header, so if auth ran
# first it would 401 the preflight before CORS could emit the
# Access-Control-Allow-* headers, blocking every browser cross-origin client
# whenever an API key is configured. Outermost CORS answers the preflight
# directly. Body-limit still runs before auth/rate-limit so oversized requests
# are rejected with 413 without consuming rate-limit tokens or touching auth
# ("reject early, reject cheap").

# Optional rate limiting (after auth in execution order)
if config.rate_limit_enabled:
    from hfl.api.middleware import RateLimitMiddleware

    app.add_middleware(
        RateLimitMiddleware,
        requests_per_window=config.rate_limit_requests,
        window_seconds=config.rate_limit_window,
    )

# API key authentication (runs before rate limiting)
app.add_middleware(APIKeyMiddleware)

# Body-size limit (runs before auth so oversized bodies are rejected
# without consuming auth/rate-limit work). 0 disables the limit.
if config.max_request_bytes > 0:
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=config.max_request_bytes)

# Request logging and metrics recording
app.add_middleware(RequestLogger)

# CORS — added LAST so it is the OUTERMOST middleware and can answer browser
# preflight (OPTIONS) requests before auth/rate-limit run. Configurable via
# config.py: ["*"] when cors_allow_all, otherwise the explicit origins.
_cors_origins = ["*"] if config.cors_allow_all else config.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=config.cors_allow_credentials,
    allow_methods=config.cors_allow_methods,
    allow_headers=config.cors_allow_headers,
)

# Register global exception handlers for HFLError hierarchy
register_exception_handlers(app)

app.include_router(anthropic_router)
app.include_router(openai_router)
app.include_router(openai_responses_router)
app.include_router(native_router)
app.include_router(batch_router)
app.include_router(blobs_router)
app.include_router(copy_router)
app.include_router(create_router)
app.include_router(embed_router)
app.include_router(ps_router)
app.include_router(pull_router)
app.include_router(push_router)
app.include_router(discover_router)
app.include_router(recommend_router)
app.include_router(smart_pull_router)
app.include_router(verify_router)
app.include_router(benchmark_router)
app.include_router(compliance_router)
app.include_router(ws_router)
app.include_router(snapshot_router)
app.include_router(lora_router)
app.include_router(draft_router)
app.include_router(show_router)
app.include_router(stop_router)
app.include_router(transcribe_router)
app.include_router(tts_router)
app.include_router(web_router)
app.include_router(health_router)
app.include_router(images_router)
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
        timeout_graceful_shutdown=30,
    )
