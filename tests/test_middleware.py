# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the api/middleware module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestRequestLogger:
    """Tests for RequestLogger middleware."""

    def test_middleware_logs_request_metadata(self):
        """Verifies that the middleware logs request metadata."""
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

            # Verify that logger.info was called with correct arguments
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args

            # Verify log format
            format_string = call_args[0][0]
            assert "method=%s" in format_string
            assert "path=%s" in format_string
            assert "status=%d" in format_string
            assert "duration=" in format_string

            # Verify positional arguments
            args = call_args[0][1:]
            assert args[0] == "GET"  # method
            assert args[1] == "/test"  # path
            assert args[2] == 200  # status

    def test_middleware_logs_post_request(self):
        """Verifies POST request logging."""
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
        """Verifies error logging."""
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
        CRITICAL: Verifies that the middleware does NOT log the request body.

        R6 - Privacy compliance: User prompts are sensitive data.
        """
        from hfl.api.middleware import RequestLogger

        app = FastAPI()
        app.add_middleware(RequestLogger)

        @app.post("/chat")
        def chat_endpoint(data: dict):
            return {"response": "ok"}

        client = TestClient(app)

        with patch("hfl.api.middleware.logger") as mock_logger:
            # Send sensitive data
            sensitive_data = {"prompt": "My secret is...", "api_key": "sk-123"}
            response = client.post("/chat", json=sensitive_data)

            # Verify that the logger call does NOT contain sensitive data
            log_call = str(mock_logger.info.call_args)
            assert "My secret" not in log_call
            assert "sk-123" not in log_call
            assert "api_key" not in log_call
            assert "prompt" not in log_call  # Field should not be logged

    def test_middleware_measures_duration(self):
        """Verifies that request duration is measured."""
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
            duration = call_args[4]  # The 5th argument is the duration
            assert duration >= 0.1  # At least 100ms
