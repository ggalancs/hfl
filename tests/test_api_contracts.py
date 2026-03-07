# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Strict API contract validation tests.

These tests verify that HFL API responses conform to the
OpenAI and Ollama API specifications.
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import reset_state
from hfl.engine.base import GenerationResult


@pytest.fixture
def client():
    """Create test client."""
    reset_state()
    return TestClient(app)


@pytest.fixture
def mock_engine():
    """Create mock engine with standard responses."""
    engine = MagicMock()
    engine.is_loaded = True
    engine.chat.return_value = GenerationResult(
        text="Hello, I'm an AI assistant.",
        tokens_prompt=10,
        tokens_generated=8,
        stop_reason="stop",
    )
    engine.generate.return_value = GenerationResult(
        text="Generated text response",
        tokens_prompt=5,
        tokens_generated=4,
        stop_reason="stop",
    )
    engine.chat_stream.return_value = iter(["Hello", ", ", "world", "!"])
    engine.generate_stream.return_value = iter(["Gen", "erated", " text"])
    return engine


@pytest.fixture
def client_with_model(client, mock_engine, temp_config, monkeypatch):
    """Client with a mock model loaded."""
    from hfl.api.state import get_state
    from hfl.models.manifest import ModelManifest
    from hfl.models.registry import ModelRegistry

    # Create and register mock model
    manifest = ModelManifest(
        name="test-model",
        repo_id="test/model",
        local_path=str(temp_config.models_dir / "test"),
        size_bytes=1000,
        format="gguf",
    )

    registry = ModelRegistry()
    registry.add(manifest)

    # Set up state with mock engine
    state = get_state()
    state.engine = mock_engine
    state.current_model = manifest

    # Mock model_loader to avoid actual loading (must be async)
    async def mock_ensure_model_loaded(model_name: str) -> None:
        pass

    monkeypatch.setattr(
        "hfl.api.routes_openai._ensure_model_loaded",
        mock_ensure_model_loaded,
    )

    return client


class TestOpenAIContract:
    """Verify strict OpenAI API compatibility."""

    REQUIRED_RESPONSE_FIELDS = {"id", "object", "created", "model", "choices"}
    REQUIRED_CHOICE_FIELDS = {"index", "message", "finish_reason"}
    REQUIRED_USAGE_FIELDS = {"prompt_tokens", "completion_tokens", "total_tokens"}

    def test_chat_completions_response_structure(self, client_with_model):
        """Every response must have required fields."""
        response = client_with_model.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check all required fields exist
        for field in self.REQUIRED_RESPONSE_FIELDS:
            assert field in data, f"Missing required field: {field}"

        # Check choices structure
        assert len(data["choices"]) > 0
        choice = data["choices"][0]
        for field in self.REQUIRED_CHOICE_FIELDS:
            assert field in choice, f"Missing choice field: {field}"

        # Check message structure
        assert "role" in choice["message"]
        assert "content" in choice["message"]
        assert choice["message"]["role"] == "assistant"

        # Check usage structure
        assert "usage" in data
        for field in self.REQUIRED_USAGE_FIELDS:
            assert field in data["usage"], f"Missing usage field: {field}"

    def test_chat_completions_id_format(self, client_with_model):
        """ID must start with 'chatcmpl-'."""
        response = client_with_model.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        data = response.json()
        assert data["id"].startswith("chatcmpl-")

    def test_chat_completions_object_type(self, client_with_model):
        """Object must be 'chat.completion' for non-streaming."""
        response = client_with_model.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

        data = response.json()
        assert data["object"] == "chat.completion"

    def test_streaming_sse_format(self, client_with_model):
        """Streaming responses must follow SSE format."""
        response = client_with_model.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Parse SSE chunks
        chunks = []
        for line in response.text.split("\n\n"):
            line = line.strip()
            if line and line.startswith("data: "):
                data_str = line[6:]
                if data_str != "[DONE]":
                    chunks.append(json.loads(data_str))

        # Verify chunk structure
        assert len(chunks) > 0
        for chunk in chunks:
            assert "id" in chunk
            assert "object" in chunk
            assert chunk["object"] == "chat.completion.chunk"
            assert "choices" in chunk
            assert len(chunk["choices"]) > 0

    def test_streaming_ends_with_done(self, client_with_model):
        """Streaming must end with [DONE] marker."""
        response = client_with_model.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert "data: [DONE]" in response.text

    def test_completions_response_structure(self, client_with_model):
        """Text completion endpoint response structure."""
        response = client_with_model.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Hello",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "id" in data
        assert data["id"].startswith("cmpl-")
        assert data["object"] == "text_completion"
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "text" in data["choices"][0]

    def test_models_list_structure(self, client_with_model):
        """Models list endpoint structure."""
        response = client_with_model.get("/v1/models")

        assert response.status_code == 200
        data = response.json()

        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)

        if len(data["data"]) > 0:
            model = data["data"][0]
            assert "id" in model
            assert "object" in model
            assert model["object"] == "model"


class TestOllamaContract:
    """Verify Ollama API compatibility."""

    def test_generate_ndjson_format(self, client_with_model):
        """Generate endpoint must return NDJSON."""
        # client_with_model fixture already mocks _ensure_model_loaded
        response = client_with_model.post(
            "/api/generate",
            json={
                "model": "test-model",
                "prompt": "Hello",
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert "application/x-ndjson" in response.headers.get("content-type", "")

        # Each line should be valid JSON
        for line in response.text.strip().split("\n"):
            if line:
                data = json.loads(line)
                assert "model" in data
                assert "done" in data

    def test_tags_response_structure(self, client_with_model):
        """Tags endpoint must return models list matching Ollama format."""
        response = client_with_model.get("/api/tags")

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert isinstance(data["models"], list)

        if len(data["models"]) > 0:
            model = data["models"][0]
            # Required fields per Ollama API
            assert "name" in model
            assert "size" in model
            assert "details" in model
            assert "modified_at" in model
            assert "digest" in model
            # Details structure
            assert "format" in model["details"]
            assert "parent_model" in model["details"]

    def test_version_response(self, client_with_model):
        """Version endpoint structure."""
        response = client_with_model.get("/api/version")

        assert response.status_code == 200
        data = response.json()
        assert "version" in data


class TestErrorResponses:
    """Test error response format consistency."""

    def test_model_not_found_404(self, client, temp_config):
        """Missing model should return 404."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 404

    def test_invalid_json_400(self, client):
        """Invalid JSON should return 400 or 422."""
        response = client.post(
            "/v1/chat/completions",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code in (400, 422)


class TestContentTypes:
    """Test Content-Type header handling."""

    def test_json_response_content_type(self, client_with_model):
        """Non-streaming should return application/json."""
        response = client_with_model.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

        assert "application/json" in response.headers.get("content-type", "")

    def test_streaming_content_type(self, client_with_model):
        """Streaming should return text/event-stream."""
        response = client_with_model.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert "text/event-stream" in response.headers.get("content-type", "")
