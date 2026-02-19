# SPDX-License-Identifier: HRUL-1.0
"""Tests for Ollama-compatible native API routes."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app, state


class TestOllamaChat:
    """Tests for Ollama /api/chat endpoint."""

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

    def test_chat_success(self):
        """Test successful chat request."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.chat.return_value = MagicMock(
            text="Hello!",
            tokens_generated=5,
            tokens_prompt=3,
            tokens_per_second=50.0,
            stop_reason="stop"
        )

        mock_model = MagicMock()
        mock_model.name = "test-model"

        state.engine = mock_engine
        state.current_model = mock_model

        client = TestClient(app)

        response = client.post(
            "/api/chat",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"]["role"] == "assistant"

    def test_chat_model_not_found(self):
        """Test chat with non-existent model."""
        state.engine = None
        state.current_model = None

        client = TestClient(app)

        with patch("hfl.api.routes_native.ModelRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.get.return_value = None
            mock_registry_class.return_value = mock_registry

            response = client.post(
                "/api/chat",
                json={
                    "model": "nonexistent-model",
                    "messages": [{"role": "user", "content": "hi"}]
                }
            )

            assert response.status_code == 404


class TestOllamaGenerate:
    """Tests for Ollama /api/generate endpoint."""

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

    def test_generate_success(self):
        """Test successful generate request."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.generate.return_value = MagicMock(
            text="World",
            tokens_generated=1,
            tokens_prompt=1,
            tokens_per_second=50.0,
            stop_reason="stop"
        )

        mock_model = MagicMock()
        mock_model.name = "test-model"

        state.engine = mock_engine
        state.current_model = mock_model

        client = TestClient(app)

        response = client.post(
            "/api/generate",
            json={
                "model": "test-model",
                "prompt": "Hello"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data


class TestOllamaTags:
    """Tests for Ollama /api/tags endpoint."""

    def test_tags_no_models(self):
        """Test tags with no models."""
        client = TestClient(app)

        with patch("hfl.api.routes_native.ModelRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.list_all.return_value = []
            mock_registry_class.return_value = mock_registry

            response = client.get("/api/tags")

            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            assert len(data["models"]) == 0

    def test_tags_with_models(self):
        """Test tags with models."""
        mock_model = MagicMock()
        mock_model.name = "test-model"
        mock_model.size_bytes = 5 * 1024 * 1024 * 1024
        mock_model.format = "GGUF"
        mock_model.quantization = "Q4_K_M"

        client = TestClient(app)

        with patch("hfl.api.routes_native.ModelRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.list_all.return_value = [mock_model]
            mock_registry_class.return_value = mock_registry

            response = client.get("/api/tags")

            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            assert len(data["models"]) == 1
            assert data["models"][0]["name"] == "test-model"


class TestOllamaVersion:
    """Tests for Ollama /api/version endpoint."""

    def test_version(self):
        """Test version endpoint."""
        client = TestClient(app)

        response = client.get("/api/version")

        assert response.status_code == 200
        data = response.json()
        assert "version" in data
