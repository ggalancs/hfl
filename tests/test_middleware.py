# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the api/middleware module."""

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestRequestLogger:
    """Tests for RequestLogger middleware."""

    def test_middleware_logs_request_metadata(self):
        """Verifies that the middleware logs request metadata via log_request."""
        from hfl.api.middleware import RequestLogger

        app = FastAPI()
        app.add_middleware(RequestLogger)

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        with patch("hfl.api.middleware.log_request") as mock_log:
            response = client.get("/test")

            assert response.status_code == 200

            # Verify that log_request was called with correct arguments
            mock_log.assert_called_once()
            call_kwargs = mock_log.call_args.kwargs

            assert call_kwargs["method"] == "GET"
            assert call_kwargs["path"] == "/test"
            assert call_kwargs["status"] == 200
            assert call_kwargs["duration_ms"] >= 0

    def test_middleware_logs_post_request(self):
        """Verifies POST request logging."""
        from hfl.api.middleware import RequestLogger

        app = FastAPI()
        app.add_middleware(RequestLogger)

        @app.post("/api/endpoint")
        def post_endpoint():
            return {"created": True}

        client = TestClient(app)

        with patch("hfl.api.middleware.log_request") as mock_log:
            response = client.post("/api/endpoint")

            assert response.status_code == 200
            call_kwargs = mock_log.call_args.kwargs
            assert call_kwargs["method"] == "POST"
            assert call_kwargs["path"] == "/api/endpoint"

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

        with patch("hfl.api.middleware.log_request") as mock_log:
            response = client.get("/error")

            assert response.status_code == 404
            call_kwargs = mock_log.call_args.kwargs
            assert call_kwargs["status"] == 404

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

        with patch("hfl.api.middleware.log_request") as mock_log:
            # Send sensitive data
            sensitive_data = {"prompt": "My secret is...", "api_key": "sk-123"}
            client.post("/chat", json=sensitive_data)

            # Verify that the log call does NOT contain sensitive data
            log_call = str(mock_log.call_args)
            assert "My secret" not in log_call
            assert "sk-123" not in log_call
            assert "api_key" not in log_call
            assert "prompt" not in log_call  # Field should not be logged

    def test_middleware_measures_duration(self):
        """Verifies that request duration is measured."""
        import time

        from hfl.api.middleware import RequestLogger

        app = FastAPI()
        app.add_middleware(RequestLogger)

        @app.get("/slow")
        def slow_endpoint():
            time.sleep(0.1)  # 100ms
            return {"status": "done"}

        client = TestClient(app)

        with patch("hfl.api.middleware.log_request") as mock_log:
            client.get("/slow")

            call_kwargs = mock_log.call_args.kwargs
            duration_ms = call_kwargs["duration_ms"]
            assert duration_ms >= 100  # At least 100ms

    def test_middleware_adds_request_id_header(self):
        """Verifies that request ID is added to response."""
        from hfl.api.middleware import RequestLogger

        app = FastAPI()
        app.add_middleware(RequestLogger)

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8  # 8 hex chars

    def test_middleware_uses_incoming_request_id(self):
        """Verifies that incoming request ID is respected."""
        from hfl.api.middleware import RequestLogger

        app = FastAPI()
        app.add_middleware(RequestLogger)

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test", headers={"X-Request-ID": "custom-id"})

        assert response.headers["X-Request-ID"] == "custom-id"
