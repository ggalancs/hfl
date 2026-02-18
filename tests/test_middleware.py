# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests para el módulo api/middleware."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestRequestLogger:
    """Tests para RequestLogger middleware."""

    def test_middleware_logs_request_metadata(self):
        """Verifica que el middleware registra metadata del request."""
        from hfl.api.middleware import RequestLogger

        app = FastAPI()
        app.add_middleware(RequestLogger)

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        with patch("hfl.api.middleware.logger") as mock_logger:
            response = client.get("/test")

            assert response.status_code == 200

            # Verificar que se llamó a logger.info con los argumentos correctos
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args

            # Verificar formato del log
            format_string = call_args[0][0]
            assert "method=%s" in format_string
            assert "path=%s" in format_string
            assert "status=%d" in format_string
            assert "duration=" in format_string

            # Verificar argumentos posicionales
            args = call_args[0][1:]
            assert args[0] == "GET"  # method
            assert args[1] == "/test"  # path
            assert args[2] == 200  # status

    def test_middleware_logs_post_request(self):
        """Verifica logging de request POST."""
        from hfl.api.middleware import RequestLogger

        app = FastAPI()
        app.add_middleware(RequestLogger)

        @app.post("/api/endpoint")
        def post_endpoint():
            return {"created": True}

        client = TestClient(app)

        with patch("hfl.api.middleware.logger") as mock_logger:
            response = client.post("/api/endpoint")

            assert response.status_code == 200
            call_args = mock_logger.info.call_args[0]
            assert call_args[1] == "POST"
            assert call_args[2] == "/api/endpoint"

    def test_middleware_logs_error_status(self):
        """Verifica logging de errores."""
        from hfl.api.middleware import RequestLogger

        app = FastAPI()
        app.add_middleware(RequestLogger)

        @app.get("/error")
        def error_endpoint():
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not found")

        client = TestClient(app)

        with patch("hfl.api.middleware.logger") as mock_logger:
            response = client.get("/error")

            assert response.status_code == 404
            call_args = mock_logger.info.call_args[0]
            assert call_args[3] == 404  # status code

    def test_middleware_privacy_no_body_logged(self):
        """
        CRÍTICO: Verifica que el middleware NO registra el body del request.

        R6 - Privacy compliance: Los prompts de usuario son datos sensibles.
        """
        from hfl.api.middleware import RequestLogger

        app = FastAPI()
        app.add_middleware(RequestLogger)

        @app.post("/chat")
        def chat_endpoint(data: dict):
            return {"response": "ok"}

        client = TestClient(app)

        with patch("hfl.api.middleware.logger") as mock_logger:
            # Enviar datos sensibles
            sensitive_data = {"prompt": "Mi secreto es...", "api_key": "sk-123"}
            response = client.post("/chat", json=sensitive_data)

            # Verificar que la llamada al logger NO contiene datos sensibles
            log_call = str(mock_logger.info.call_args)
            assert "Mi secreto" not in log_call
            assert "sk-123" not in log_call
            assert "api_key" not in log_call
            assert "prompt" not in log_call  # No debería loguearse el campo

    def test_middleware_measures_duration(self):
        """Verifica que mide la duración del request."""
        from hfl.api.middleware import RequestLogger
        import time

        app = FastAPI()
        app.add_middleware(RequestLogger)

        @app.get("/slow")
        def slow_endpoint():
            time.sleep(0.1)  # 100ms
            return {"status": "done"}

        client = TestClient(app)

        with patch("hfl.api.middleware.logger") as mock_logger:
            client.get("/slow")

            call_args = mock_logger.info.call_args[0]
            duration = call_args[4]  # El 5to argumento es la duración
            assert duration >= 0.1  # Al menos 100ms
