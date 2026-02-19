# SPDX-License-Identifier: HRUL-1.0
"""Edge case tests for OpenAI-compatible API routes."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app, state
from hfl.api.routes_openai import _to_gen_config, _ensure_model_loaded


class TestToGenConfigEdgeCases:
    """Edge case tests for _to_gen_config function."""

    def test_stop_as_empty_list(self):
        """Test stop as empty list."""
        req = MagicMock()
        req.stop = []
        req.temperature = 0.7
        req.top_p = 0.9
        req.max_tokens = 100
        req.seed = 42

        config = _to_gen_config(req)

        assert config.stop == []

    def test_stop_as_none(self):
        """Test stop as None."""
        req = MagicMock()
        req.stop = None
        req.temperature = 0.7
        req.top_p = 0.9
        req.max_tokens = 100
        req.seed = 42

        config = _to_gen_config(req)

        assert config.stop is None

    def test_max_tokens_none_uses_default(self):
        """Test max_tokens None uses default 2048."""
        req = MagicMock()
        req.stop = None
        req.temperature = 0.7
        req.top_p = 0.9
        req.max_tokens = None
        req.seed = 42

        config = _to_gen_config(req)

        assert config.max_tokens == 2048

    def test_seed_none_uses_default(self):
        """Test seed None uses default -1."""
        req = MagicMock()
        req.stop = None
        req.temperature = 0.7
        req.top_p = 0.9
        req.max_tokens = 100
        req.seed = None

        config = _to_gen_config(req)

        assert config.seed == -1

    def test_all_parameters_set(self):
        """Test all parameters are properly set."""
        req = MagicMock()
        req.stop = ["END", "STOP"]
        req.temperature = 0.5
        req.top_p = 0.8
        req.max_tokens = 500
        req.seed = 123

        config = _to_gen_config(req)

        assert config.stop == ["END", "STOP"]
        assert config.temperature == 0.5
        assert config.top_p == 0.8
        assert config.max_tokens == 500
        assert config.seed == 123


class TestEnsureModelLoaded:
    """Tests for _ensure_model_loaded function."""

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

    def test_same_model_no_reload(self):
        """Test same model doesn't trigger reload."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True

        mock_model = MagicMock()
        mock_model.name = "test-model"

        state.engine = mock_engine
        state.current_model = mock_model

        # Should return early without doing anything
        _ensure_model_loaded("test-model")

        # Engine should not be unloaded
        mock_engine.unload.assert_not_called()

    def test_different_model_unloads_current(self):
        """Test different model unloads the current one."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True

        mock_current_model = MagicMock()
        mock_current_model.name = "model-a"

        state.engine = mock_engine
        state.current_model = mock_current_model

        # Mock the registry to return None (model not found)
        with patch("hfl.api.routes_openai.ModelRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.get.return_value = None
            mock_registry_class.return_value = mock_registry

            with pytest.raises(Exception):  # HTTPException
                _ensure_model_loaded("model-b")

            # Engine should be unloaded before trying to load new model
            mock_engine.unload.assert_called_once()


class TestChatCompletionsStreamingEdgeCases:
    """Edge case tests for streaming chat completions."""

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

    def test_chat_completions_with_system_message(self):
        """Test chat completions with system message."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.chat.return_value = MagicMock(
            text="I'm a helpful assistant!",
            tokens_generated=5,
            tokens_prompt=10,
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
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"}
                ]
            }
        )

        assert response.status_code == 200

    def test_chat_completions_multi_turn(self):
        """Test chat completions with multi-turn conversation."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.chat.return_value = MagicMock(
            text="You're welcome!",
            tokens_generated=3,
            tokens_prompt=20,
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
                "messages": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "Thanks!"}
                ]
            }
        )

        assert response.status_code == 200


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_check(self):
        """Test health endpoint returns healthy."""
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"
