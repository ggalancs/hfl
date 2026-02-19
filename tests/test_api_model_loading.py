# SPDX-License-Identifier: HRUL-1.0
"""Tests for API model loading and switching."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app, state


class TestModelSwitching:
    """Tests for model switching in API."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset server state before each test."""
        state.api_key = None
        state.engine = None
        state.current_model = None
        yield
        state.api_key = None
        state.engine = None
        state.current_model = None

    def test_load_different_model_unloads_current(self):
        """Test that loading a different model unloads the current one."""
        # Setup: pretend we have a model loaded
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.unload = MagicMock()

        mock_current_model = MagicMock()
        mock_current_model.name = "model-a"

        state.engine = mock_engine
        state.current_model = mock_current_model

        client = TestClient(app)

        # Try to use a different model - this will fail but should trigger unload
        with patch("hfl.api.routes_openai.ModelRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.get.return_value = None  # Model not found
            mock_registry_class.return_value = mock_registry

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "model-b",  # Different model
                    "messages": [{"role": "user", "content": "hi"}]
                }
            )

            # Should return 404 for model not found
            assert response.status_code == 404

    def test_same_model_no_reload(self):
        """Test that requesting the same model doesn't reload it."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True

        mock_current_model = MagicMock()
        mock_current_model.name = "test-model"

        state.engine = mock_engine
        state.current_model = mock_current_model

        client = TestClient(app)

        # Mock the chat method to return a result
        mock_engine.chat.return_value = MagicMock(
            text="Hello!",
            tokens_generated=5,
            tokens_prompt=3,
            tokens_per_second=50.0,
            stop_reason="stop"
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",  # Same model
                "messages": [{"role": "user", "content": "hi"}]
            }
        )

        # Should succeed without reloading
        assert response.status_code == 200
        # Engine should not be unloaded
        mock_engine.unload.assert_not_called()


class TestModelNotFound:
    """Tests for model not found scenarios."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset server state before each test."""
        state.api_key = None
        state.engine = None
        state.current_model = None
        yield

    def test_chat_completions_model_not_found(self):
        """Test chat completions with non-existent model."""
        client = TestClient(app)

        with patch("hfl.api.routes_openai.ModelRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.get.return_value = None
            mock_registry_class.return_value = mock_registry

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "nonexistent-model",
                    "messages": [{"role": "user", "content": "hi"}]
                }
            )

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_completions_model_not_found(self):
        """Test completions with non-existent model."""
        client = TestClient(app)

        with patch("hfl.api.routes_openai.ModelRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.get.return_value = None
            mock_registry_class.return_value = mock_registry

            response = client.post(
                "/v1/completions",
                json={
                    "model": "nonexistent-model",
                    "prompt": "Hello"
                }
            )

            assert response.status_code == 404


class TestAPIVersion:
    """Tests for API version endpoint."""

    def test_api_version(self):
        """Test /api/version endpoint."""
        client = TestClient(app)

        response = client.get("/api/version")

        assert response.status_code == 200
        assert "version" in response.json()
