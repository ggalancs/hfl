# SPDX-License-Identifier: HRUL-1.0
"""Tests for Anthropic Messages API-compatible routes."""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state


def _make_engine(text="Hello!", tokens_generated=5, tokens_prompt=10):
    """Create a mock engine with standard chat response."""
    engine = MagicMock()
    engine.is_loaded = True
    engine.chat.return_value = MagicMock(
        text=text,
        tokens_generated=tokens_generated,
        tokens_prompt=tokens_prompt,
        tokens_per_second=50.0,
        stop_reason="stop",
    )
    return engine


def _make_model(name="qwen-coder"):
    """Create a mock model manifest."""
    model = MagicMock()
    model.name = name
    return model


class TestAnthropicMessages:
    """Tests for POST /v1/messages endpoint."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset server state before each test."""
        state = get_state()
        state.api_key = None
        state.engine = None
        state.current_model = None
        yield
        state.api_key = None
        state.engine = None
        state.current_model = None

    def test_basic_message(self):
        """Test basic non-streaming message creation."""
        engine = _make_engine()
        state = get_state()
        state.engine = engine
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "qwen-coder",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert len(data["content"]) == 1
        assert data["content"][0]["type"] == "text"
        assert data["content"][0]["text"] == "Hello!"
        assert data["stop_reason"] == "end_turn"
        assert "input_tokens" in data["usage"]
        assert "output_tokens" in data["usage"]
        assert data["id"].startswith("msg_")

    def test_model_prefix_stripped(self):
        """Test that hfl/ prefix is stripped from model name."""
        engine = _make_engine()
        state = get_state()
        state.engine = engine
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "hfl/qwen-coder",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        # The response should preserve the original model name
        data = response.json()
        assert data["model"] == "hfl/qwen-coder"

    def test_system_prompt_string(self):
        """Test system prompt as string is passed to engine."""
        engine = _make_engine()
        state = get_state()
        state.engine = engine
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "qwen-coder",
                "max_tokens": 1024,
                "system": "You are a helpful coding assistant.",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        # Verify system prompt was passed as first message
        call_args = engine.chat.call_args[0]
        messages = call_args[0]
        assert messages[0].role == "system"
        assert messages[0].content == "You are a helpful coding assistant."
        assert messages[1].role == "user"

    def test_system_prompt_content_blocks(self):
        """Test system prompt as list of content blocks."""
        engine = _make_engine()
        state = get_state()
        state.engine = engine
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "qwen-coder",
                "max_tokens": 1024,
                "system": [{"type": "text", "text": "You are helpful."}],
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        call_args = engine.chat.call_args[0]
        messages = call_args[0]
        assert messages[0].role == "system"
        assert messages[0].content == "You are helpful."

    def test_content_blocks_in_messages(self):
        """Test message content as list of content blocks."""
        engine = _make_engine()
        state = get_state()
        state.engine = engine
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "qwen-coder",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Hello "},
                            {"type": "text", "text": "world"},
                        ],
                    }
                ],
            },
        )

        assert response.status_code == 200
        call_args = engine.chat.call_args[0]
        messages = call_args[0]
        assert messages[0].content == "Hello world"

    def test_stop_sequences_passed(self):
        """Test that stop_sequences are passed to generation config."""
        engine = _make_engine()
        state = get_state()
        state.engine = engine
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "qwen-coder",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
                "stop_sequences": ["\n\nHuman:"],
            },
        )

        assert response.status_code == 200
        call_args = engine.chat.call_args[0]
        config = call_args[1]
        assert config.stop == ["\n\nHuman:"]

    def test_temperature_and_top_p(self):
        """Test temperature and top_p parameters."""
        engine = _make_engine()
        state = get_state()
        state.engine = engine
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "qwen-coder",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.5,
                "top_p": 0.8,
            },
        )

        assert response.status_code == 200
        call_args = engine.chat.call_args[0]
        config = call_args[1]
        assert config.temperature == 0.5
        assert config.top_p == 0.8

    def test_stop_reason_mapping(self):
        """Test that stop_reason 'stop' maps to 'end_turn'."""
        engine = _make_engine()
        state = get_state()
        state.engine = engine
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "qwen-coder",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        data = response.json()
        assert data["stop_reason"] == "end_turn"

    def test_stop_reason_max_tokens(self):
        """Test that stop_reason 'length' maps to 'max_tokens'."""
        engine = MagicMock()
        engine.is_loaded = True
        engine.chat.return_value = MagicMock(
            text="partial",
            tokens_generated=100,
            tokens_prompt=10,
            tokens_per_second=50.0,
            stop_reason="length",
        )
        state = get_state()
        state.engine = engine
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "qwen-coder",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        data = response.json()
        assert data["stop_reason"] == "max_tokens"

    def test_empty_messages_rejected(self):
        """Test that empty messages list is rejected."""
        state = get_state()
        state.engine = _make_engine()
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "qwen-coder",
                "max_tokens": 1024,
                "messages": [],
            },
        )

        assert response.status_code == 422

    def test_model_not_loaded(self):
        """Test response when no model is loaded and model not found."""
        state = get_state()
        state.engine = None
        state.current_model = None

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "nonexistent-model",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        # Should get 404 (model not found) or similar error
        assert response.status_code in (400, 404, 503)

    def test_context_window_exceeded_error(self):
        """Test proper error when tokens exceed context window."""
        engine = MagicMock()
        engine.is_loaded = True
        engine.chat.side_effect = ValueError(
            "Requested tokens (5705) exceed context window of 4096"
        )

        state = get_state()
        state.engine = engine
        state.current_model = _make_model()

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/messages",
            json={
                "model": "qwen-coder",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "invalid_request_error"
        assert "exceed context window" in data["error"]["message"]

    def test_disclaimer_header_present(self):
        """Test that X-AI-Disclaimer header is added for /v1/messages."""
        engine = _make_engine()
        state = get_state()
        state.engine = engine
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "qwen-coder",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        assert "X-AI-Disclaimer" in response.headers


class TestAnthropicMessagesStreaming:
    """Tests for streaming POST /v1/messages endpoint."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset server state before each test."""
        state = get_state()
        state.api_key = None
        state.engine = None
        state.current_model = None
        yield
        state.api_key = None
        state.engine = None
        state.current_model = None

    def test_streaming_response_format(self):
        """Test streaming response follows Anthropic SSE format."""
        engine = MagicMock()
        engine.is_loaded = True
        engine.chat_stream.return_value = iter(["Hello", " world", "!"])

        state = get_state()
        state.engine = engine
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "qwen-coder",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        text = response.text
        # Should contain all required SSE events
        assert "event: message_start" in text
        assert "event: content_block_start" in text
        assert "event: ping" in text
        assert "event: content_block_delta" in text
        assert "event: content_block_stop" in text
        assert "event: message_delta" in text
        assert "event: message_stop" in text

    def test_streaming_content_deltas(self):
        """Test that content deltas contain the correct tokens."""
        engine = MagicMock()
        engine.is_loaded = True
        engine.chat_stream.return_value = iter(["foo", "bar"])

        state = get_state()
        state.engine = engine
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "qwen-coder",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        # Parse SSE events
        lines = response.text.split("\n")
        deltas = []
        for i, line in enumerate(lines):
            if line.startswith("data: ") and i > 0 and lines[i - 1] == "event: content_block_delta":
                data = json.loads(line[6:])
                deltas.append(data["delta"]["text"])

        assert deltas == ["foo", "bar"]

    def test_streaming_message_start_structure(self):
        """Test that message_start event has correct structure."""
        engine = MagicMock()
        engine.is_loaded = True
        engine.chat_stream.return_value = iter(["x"])

        state = get_state()
        state.engine = engine
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "qwen-coder",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        lines = response.text.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("data: ") and i > 0 and lines[i - 1] == "event: message_start":
                data = json.loads(line[6:])
                msg = data["message"]
                assert msg["type"] == "message"
                assert msg["role"] == "assistant"
                assert msg["content"] == []
                assert msg["stop_reason"] is None
                assert "usage" in msg
                assert msg["id"].startswith("msg_")
                break
        else:
            pytest.fail("No message_start event found")

    def test_streaming_message_delta_has_stop_reason(self):
        """Test that message_delta has stop_reason and usage."""
        engine = MagicMock()
        engine.is_loaded = True
        engine.chat_stream.return_value = iter(["done"])

        state = get_state()
        state.engine = engine
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "qwen-coder",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        lines = response.text.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("data: ") and i > 0 and lines[i - 1] == "event: message_delta":
                data = json.loads(line[6:])
                assert data["delta"]["stop_reason"] == "end_turn"
                assert "output_tokens" in data["usage"]
                break
        else:
            pytest.fail("No message_delta event found")


class TestAnthropicSchemas:
    """Tests for Anthropic schema validation and helpers."""

    def test_resolve_model_name_with_prefix(self):
        """Test model name resolution strips hfl/ prefix."""
        from hfl.api.schemas.anthropic import AnthropicMessagesRequest

        req = AnthropicMessagesRequest(
            model="hfl/qwen-coder",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert req.resolve_model_name() == "qwen-coder"

    def test_resolve_model_name_without_prefix(self):
        """Test model name without prefix is returned as-is."""
        from hfl.api.schemas.anthropic import AnthropicMessagesRequest

        req = AnthropicMessagesRequest(
            model="qwen-coder",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert req.resolve_model_name() == "qwen-coder"

    def test_resolve_model_name_other_prefix(self):
        """Test model name with other prefix (e.g., org/model)."""
        from hfl.api.schemas.anthropic import AnthropicMessagesRequest

        req = AnthropicMessagesRequest(
            model="myorg/my-model",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert req.resolve_model_name() == "my-model"

    def test_get_system_text_string(self):
        """Test system text extraction from string."""
        from hfl.api.schemas.anthropic import AnthropicMessagesRequest

        req = AnthropicMessagesRequest(
            model="test",
            max_tokens=1024,
            system="Be helpful",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert req.get_system_text() == "Be helpful"

    def test_get_system_text_blocks(self):
        """Test system text extraction from content blocks."""
        from hfl.api.schemas.anthropic import AnthropicMessagesRequest

        req = AnthropicMessagesRequest(
            model="test",
            max_tokens=1024,
            system=[
                {"type": "text", "text": "Part 1. "},
                {"type": "text", "text": "Part 2."},
            ],
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert req.get_system_text() == "Part 1. Part 2."

    def test_get_system_text_none(self):
        """Test system text extraction when no system prompt."""
        from hfl.api.schemas.anthropic import AnthropicMessagesRequest

        req = AnthropicMessagesRequest(
            model="test",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert req.get_system_text() is None

    def test_message_get_text_string(self):
        """Test message text extraction from string content."""
        from hfl.api.schemas.anthropic import AnthropicMessage

        msg = AnthropicMessage(role="user", content="Hello")
        assert msg.get_text() == "Hello"

    def test_message_get_text_blocks(self):
        """Test message text extraction from content blocks."""
        from hfl.api.schemas.anthropic import AnthropicMessage

        msg = AnthropicMessage(
            role="user",
            content=[
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ],
        )
        assert msg.get_text() == "Hello world"


class TestAnthropicConverter:
    """Tests for Anthropic to GenerationConfig converter."""

    def test_default_values(self):
        """Test converter uses defaults when optional fields are None."""
        from hfl.api.converters import anthropic_to_generation_config
        from hfl.api.schemas.anthropic import AnthropicMessagesRequest

        req = AnthropicMessagesRequest(
            model="test",
            max_tokens=2048,
            messages=[{"role": "user", "content": "Hi"}],
        )
        config = anthropic_to_generation_config(req)
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.max_tokens == 2048
        assert config.stop is None

    def test_explicit_values(self):
        """Test converter respects explicit parameter values."""
        from hfl.api.converters import anthropic_to_generation_config
        from hfl.api.schemas.anthropic import AnthropicMessagesRequest

        req = AnthropicMessagesRequest(
            model="test",
            max_tokens=512,
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.3,
            top_p=0.5,
            top_k=10,
            stop_sequences=["\nHuman:"],
        )
        config = anthropic_to_generation_config(req)
        assert config.temperature == 0.3
        assert config.top_p == 0.5
        assert config.top_k == 10
        assert config.max_tokens == 512
        assert config.stop == ["\nHuman:"]


class TestAnthropicAuth:
    """Tests for API key authentication with Anthropic endpoints."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        state = get_state()
        state.api_key = None
        state.engine = None
        state.current_model = None
        yield
        state.api_key = None
        state.engine = None
        state.current_model = None

    def test_x_api_key_header_auth(self):
        """Test that x-api-key header works (Anthropic SDK sends this)."""
        engine = _make_engine()
        state = get_state()
        state.api_key = "hfl"
        state.engine = engine
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            headers={"X-API-Key": "hfl"},
            json={
                "model": "qwen-coder",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200

    def test_bearer_token_auth(self):
        """Test that Bearer token auth also works."""
        engine = _make_engine()
        state = get_state()
        state.api_key = "hfl"
        state.engine = engine
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer hfl"},
            json={
                "model": "qwen-coder",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200

    def test_wrong_api_key_rejected(self):
        """Test that wrong API key is rejected."""
        state = get_state()
        state.api_key = "hfl"
        state.engine = _make_engine()
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            headers={"X-API-Key": "wrong"},
            json={
                "model": "qwen-coder",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 401

    def test_missing_api_key_rejected(self):
        """Test that missing API key is rejected when configured."""
        state = get_state()
        state.api_key = "hfl"
        state.engine = _make_engine()
        state.current_model = _make_model()

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "qwen-coder",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 401
