# SPDX-License-Identifier: HRUL-1.0
"""Tests for API server authentication and middleware."""

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state


class TestAPIKeyMiddleware:
    """Tests for API key authentication middleware."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset server state before each test."""
        get_state().api_key = None
        get_state().engine = None
        get_state().current_model = None
        yield
        get_state().api_key = None

    def test_no_auth_required_when_no_key_configured(self):
        """Test that requests pass when no API key is configured."""
        get_state().api_key = None
        client = TestClient(app)

        response = client.get("/v1/models")

        assert response.status_code == 200

    def test_public_endpoints_bypass_auth(self):
        """Test that public endpoints don't require authentication."""
        get_state().api_key = "test-secret-key"
        client = TestClient(app)

        # Root endpoint
        response = client.get("/")
        assert response.status_code == 200

        # Health endpoint
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_subpaths_bypass_auth(self):
        """Test that all health subpaths bypass authentication."""
        get_state().api_key = "test-secret-key"
        client = TestClient(app)

        # All health endpoints should work without auth
        health_paths = ["/health", "/health/ready", "/health/live", "/health/deep"]

        for path in health_paths:
            response = client.get(path)
            assert response.status_code == 200, f"{path} should bypass auth"

    def test_health_endpoints_bypass_auth_with_any_prefix(self):
        """Test that health prefix matching works correctly."""
        get_state().api_key = "test-secret-key"
        client = TestClient(app)

        # This should work (health prefix)
        response = client.get("/health")
        assert response.status_code == 200

        # Non-health paths should still require auth
        response = client.get("/v1/models")
        assert response.status_code == 401

    def test_auth_required_with_api_key_configured(self):
        """Test that authentication is required when API key is set."""
        get_state().api_key = "test-secret-key"
        client = TestClient(app)

        response = client.get("/v1/models")

        assert response.status_code == 401
        assert "Invalid or missing API key" in response.json()["error"]

    def test_bearer_token_authentication(self):
        """Test authentication with Bearer token."""
        get_state().api_key = "test-secret-key"
        client = TestClient(app)

        response = client.get("/v1/models", headers={"Authorization": "Bearer test-secret-key"})

        assert response.status_code == 200

    def test_x_api_key_header_authentication(self):
        """Test authentication with X-API-Key header."""
        get_state().api_key = "test-secret-key"
        client = TestClient(app)

        response = client.get("/v1/models", headers={"X-API-Key": "test-secret-key"})

        assert response.status_code == 200

    def test_invalid_bearer_token_rejected(self):
        """Test that invalid Bearer token is rejected."""
        get_state().api_key = "test-secret-key"
        client = TestClient(app)

        response = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})

        assert response.status_code == 401

    def test_invalid_x_api_key_rejected(self):
        """Test that invalid X-API-Key is rejected."""
        get_state().api_key = "test-secret-key"
        client = TestClient(app)

        response = client.get("/v1/models", headers={"X-API-Key": "wrong-key"})

        assert response.status_code == 401

    def test_www_authenticate_header_on_401(self):
        """Test that WWW-Authenticate header is set on 401."""
        get_state().api_key = "test-secret-key"
        client = TestClient(app)

        response = client.get("/v1/models")

        assert response.status_code == 401
        assert response.headers.get("WWW-Authenticate") == "Bearer"

    def test_malformed_bearer_token_rejected(self):
        """Test that malformed Bearer token is rejected."""
        get_state().api_key = "test-secret-key"
        client = TestClient(app)

        response = client.get("/v1/models", headers={"Authorization": "NotBearer test-secret-key"})

        assert response.status_code == 401


class TestDisclaimerMiddleware:
    """Tests for AI disclaimer middleware."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset server state before each test."""
        get_state().api_key = None
        get_state().engine = None
        get_state().current_model = None
        yield

    def test_disclaimer_header_on_chat_completions(self):
        """Test that disclaimer header is added to chat completions."""
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        )

        # Even on error, the disclaimer should be present
        assert "X-AI-Disclaimer" in response.headers

    def test_disclaimer_header_on_completions(self):
        """Test that disclaimer header is added to completions."""
        client = TestClient(app)

        response = client.post("/v1/completions", json={"model": "test", "prompt": "hi"})

        assert "X-AI-Disclaimer" in response.headers

    def test_disclaimer_header_on_api_generate(self):
        """Test that disclaimer header is added to /api/generate."""
        client = TestClient(app)

        response = client.post("/api/generate", json={"model": "test", "prompt": "hi"})

        assert "X-AI-Disclaimer" in response.headers

    def test_disclaimer_header_on_api_chat(self):
        """Test that disclaimer header is added to /api/chat."""
        client = TestClient(app)

        response = client.post(
            "/api/chat", json={"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        )

        assert "X-AI-Disclaimer" in response.headers

    def test_no_disclaimer_on_non_ai_endpoints(self):
        """Test that non-AI endpoints don't get disclaimer."""
        client = TestClient(app)

        response = client.get("/v1/models")

        assert "X-AI-Disclaimer" not in response.headers

    def test_disclaimer_content(self):
        """Test the content of the disclaimer header."""
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        )

        disclaimer = response.headers.get("X-AI-Disclaimer", "")
        assert "AI-generated" in disclaimer
        assert "inaccurate" in disclaimer or "inappropriate" in disclaimer


class TestServerState:
    """Tests for server state management."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset server state before each test."""
        get_state().api_key = None
        get_state().engine = None
        get_state().current_model = None
        yield

    def test_initial_state(self):
        """Test initial server state values."""
        assert get_state().engine is None
        assert get_state().current_model is None
        assert get_state().api_key is None

    def test_state_api_key_assignment(self):
        """Test that API key can be assigned."""
        get_state().api_key = "new-key"
        assert get_state().api_key == "new-key"


class TestHealthEndpoint:
    """Tests for health endpoint with different states."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset server state before each test."""
        get_state().api_key = None
        get_state().engine = None
        get_state().current_model = None
        yield

    def test_health_no_model(self):
        """Test health endpoint when no model is loaded."""
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is False
        assert data["current_model"] is None

    def test_health_with_model(self):
        """Test health endpoint when a model is loaded."""
        from unittest.mock import MagicMock

        mock_engine = MagicMock()
        mock_engine.is_loaded = True

        mock_model = MagicMock()
        mock_model.name = "test-model"

        get_state().engine = mock_engine
        get_state().current_model = mock_model

        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["current_model"] == "test-model"

        # Cleanup
        get_state().engine = None
        get_state().current_model = None
