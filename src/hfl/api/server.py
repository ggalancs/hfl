# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Servidor API REST compatible con OpenAI y Ollama.

Endpoints implementados:
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

Cumplimiento Legal (R9 - Auditoría):
- Disclaimer header en todas las respuestas AI
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from hfl.api.routes_native import router as native_router
from hfl.api.routes_openai import router as openai_router
from hfl.config import config


# Estado global del servidor
class ServerState:
    engine = None  # InferenceEngine activo
    current_model = None  # ModelManifest del modelo cargado


state = ServerState()


# R9 - Disclaimer Middleware (Auditoría Legal)
# Añade header de exención de responsabilidad a todas las respuestas AI
class DisclaimerMiddleware(BaseHTTPMiddleware):
    """Middleware que añade disclaimer a respuestas de endpoints AI."""

    AI_ENDPOINTS = {"/v1/chat/completions", "/v1/completions", "/api/generate", "/api/chat"}

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Solo añadir disclaimer a endpoints de generación
        if request.url.path in self.AI_ENDPOINTS:
            response.headers["X-AI-Disclaimer"] = (
                "AI-generated content. May be inaccurate or inappropriate. "
                "User assumes all responsibility for use of outputs."
            )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle del servidor."""
    yield
    # Cleanup al cerrar
    if state.engine and state.engine.is_loaded:
        state.engine.unload()


app = FastAPI(
    title="hfl API",
    description="API compatible con OpenAI y Ollama para modelos HuggingFace",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# R9 - Añadir disclaimer middleware
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
