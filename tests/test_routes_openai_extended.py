# SPDX-License-Identifier: HRUL-1.0
"""Extended tests for OpenAI-compatible API routes."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app, state


class TestChatCompletionsExtended:
    """Extended tests for chat completions endpoint."""

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

    def test_chat_completions_with_stop_sequence_list(self):
        """Test chat completions with stop sequences as list."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.chat.return_value = MagicMock(
            text="Hello",
            tokens_generated=1,
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
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stop": ["END", "STOP"]
            }
        )

        assert response.status_code == 200
        # Verify stop sequences were passed to config
        call_args = mock_engine.chat.call_args
        config = call_args[1].get("config") or call_args[0][1]
        assert config.stop == ["END", "STOP"]

    def test_chat_completions_with_single_stop(self):
        """Test chat completions with single stop sequence (not list)."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.chat.return_value = MagicMock(
            text="Hello",
            tokens_generated=1,
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
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stop": "END"
            }
        )

        assert response.status_code == 200
        call_args = mock_engine.chat.call_args
        config = call_args[1].get("config") or call_args[0][1]
        assert config.stop == ["END"]

    def test_chat_completions_no_stop(self):
        """Test chat completions without stop sequences."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.chat.return_value = MagicMock(
            text="Hello!",
            tokens_generated=2,
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
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}]
            }
        )

        assert response.status_code == 200


class TestCompletionsExtended:
    """Extended tests for completions endpoint."""

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

    def test_completions_with_temperature(self):
        """Test completions with custom temperature."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.generate.return_value = MagicMock(
            text="world",
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
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Hello",
                "temperature": 0.5,
                "top_p": 0.8
            }
        )

        assert response.status_code == 200
        call_args = mock_engine.generate.call_args
        config = call_args[1].get("config") or call_args[0][1]
        assert config.temperature == 0.5
        assert config.top_p == 0.8

    def test_completions_with_max_tokens(self):
        """Test completions with custom max_tokens."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.generate.return_value = MagicMock(
            text="world",
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
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Hello",
                "max_tokens": 100
            }
        )

        assert response.status_code == 200
        call_args = mock_engine.generate.call_args
        config = call_args[1].get("config") or call_args[0][1]
        assert config.max_tokens == 100


class TestModelsEndpoint:
    """Tests for /v1/models endpoint."""

    def test_list_models_empty(self):
        """Test listing models when none available."""
        client = TestClient(app)

        with patch("hfl.api.routes_openai.ModelRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.list_all.return_value = []
            mock_registry_class.return_value = mock_registry

            response = client.get("/v1/models")

            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert len(data["data"]) == 0

    def test_list_models_with_models(self):
        """Test listing models when models are available."""
        mock_model = MagicMock()
        mock_model.name = "test-model"
        mock_model.created_at = "2024-01-01T00:00:00Z"

        client = TestClient(app)

        with patch("hfl.api.routes_openai.ModelRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.list_all.return_value = [mock_model]
            mock_registry_class.return_value = mock_registry

            response = client.get("/v1/models")

            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "test-model"
