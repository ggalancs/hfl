# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Middleware para logging, CORS y manejo de errores.

PRIVACIDAD (R6 - Auditoría Legal):
Este middleware implementa logging "privacy-safe" que:
- NUNCA registra el contenido de los requests (prompts)
- NUNCA registra el contenido de las respuestas (outputs AI)
- NUNCA registra headers de autenticación
- Solo registra metadata: método, path, status, duración
"""

import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("hfl")


class RequestLogger(BaseHTTPMiddleware):
    """
    Middleware de logging privacy-safe.

    IMPORTANTE: Este logger está diseñado para NUNCA registrar:
    - Request bodies (contienen prompts del usuario)
    - Response bodies (contienen outputs del modelo)
    - Authorization headers (contienen tokens)
    - User-Agent u otros headers identificadores

    Solo se registra metadata básica para debugging y métricas.
    """

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        # Solo loguear metadata, NUNCA el body ni headers sensibles
        # R6 - Privacy compliance: no personal data in logs
        logger.info(
            "method=%s path=%s status=%d duration=%.3fs",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            # NO incluir: request body, headers auth, user-agent, IP real
        )
        return response
