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
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from hfl.api.routes_native import router as native_router
from hfl.api.routes_openai import router as openai_router
from hfl.config import config


# Global server state
class ServerState:
    engine = None  # Active InferenceEngine
    current_model = None  # ModelManifest of loaded model


state = ServerState()


# R9 - Disclaimer Middleware (Legal Audit)
# Adds disclaimer header to all AI responses
class DisclaimerMiddleware(BaseHTTPMiddleware):
    """Middleware that adds disclaimer to AI endpoint responses."""

    AI_ENDPOINTS = {"/v1/chat/completions", "/v1/completions", "/api/generate", "/api/chat"}

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Only add disclaimer to generation endpoints
        if request.url.path in self.AI_ENDPOINTS:
            response.headers["X-AI-Disclaimer"] = (
                "AI-generated content. May be inaccurate or inappropriate. "
                "User assumes all responsibility for use of outputs."
            )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Server lifecycle."""
    yield
    # Cleanup on shutdown
    if state.engine and state.engine.is_loaded:
        state.engine.unload()


app = FastAPI(
    title="hfl API",
    description="OpenAI and Ollama compatible API for HuggingFace models",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# R9 - Add disclaimer middleware
app.add_middleware(DisclaimerMiddleware)

app.include_router(openai_router)
app.include_router(native_router)


@app.get("/")
async def root():
    return {"status": "hfl is running"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": state.engine is not None and state.engine.is_loaded,
        "current_model": state.current_model.name if state.current_model else None,
    }


def start_server(host: str | None = None, port: int | None = None):
    uvicorn.run(
        app,
        host=host or config.host,
        port=port or config.port,
        log_level="info",
    )
