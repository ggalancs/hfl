# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""OpenAI API compatibility tests.

Validates that our API responses match the OpenAI API specification.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.engine.base import GenerationResult


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_loaded_state():
    """Mock server state with a loaded model."""
    mock_engine = MagicMock()
    mock_engine.is_loaded = True

    mock_manifest = MagicMock()
    mock_manifest.name = "test-model"
    mock_manifest.repo_id = "test-org/test-model"

    mock_state = MagicMock()
    mock_state.engine = mock_engine
    mock_state.current_model = mock_manifest
    mock_state.api_key = None

    return mock_state, mock_engine


class TestChatCompletionsFormat:
    """Tests for /v1/chat/completions response format."""

    def test_chat_completion_response_structure(self, client, mock_loaded_state):
        """Non-streaming response should match OpenAI format."""
        mock_state, mock_engine = mock_loaded_state
        mock_engine.chat.return_value = GenerationResult(
            text="Hello! How can I help you?",
            tokens_prompt=10,
            tokens_generated=7,
            stop_reason="stop",
        )

        with patch("hfl.api.routes_openai._get_state", return_value=mock_state):
            with patch("hfl.api.routes_openai._ensure_model_loaded"):
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": False,
                    },
                )

        assert response.status_code == 200
        data = response.json()

        # Required fields per OpenAI spec
        assert "id" in data
        assert data["id"].startswith("chatcmpl-")
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert isinstance(data["created"], int)
        assert data["model"] == "test-model"

        # Choices array
        assert "choices" in data
        assert isinstance(data["choices"], list)
        assert len(data["choices"]) >= 1

        choice = data["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert "finish_reason" in choice

        message = choice["message"]
        assert "role" in message
        assert message["role"] == "assistant"
        assert "content" in message

        # Usage
        assert "usage" in data
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_chat_completion_streaming_format(self, client, mock_loaded_state):
        """Streaming response should match OpenAI SSE format."""
        mock_state, mock_engine = mock_loaded_state
        mock_engine.chat_stream.return_value = iter(["Hello", "!", " How", " can", " I", " help", "?"])

        with patch("hfl.api.routes_openai._get_state", return_value=mock_state):
            with patch("hfl.api.routes_openai._ensure_model_loaded"):
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": True,
                    },
                )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE events
        events = []
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                data = line[6:]
                if data != "[DONE]":
                    events.append(json.loads(data))

        # Verify chunk format
        for event in events:
            assert "id" in event
            assert event["object"] == "chat.completion.chunk"
            assert "created" in event
            assert "model" in event
            assert "choices" in event

            for choice in event["choices"]:
                assert "index" in choice
                assert "delta" in choice
                assert "finish_reason" in choice

        # Last chunk should have finish_reason
        assert events[-1]["choices"][0]["finish_reason"] == "stop"


class TestCompletionsFormat:
    """Tests for /v1/completions response format."""

    def test_completion_response_structure(self, client, mock_loaded_state):
        """Text completion should match OpenAI format."""
        mock_state, mock_engine = mock_loaded_state
        mock_engine.generate.return_value = GenerationResult(
            text="The capital of France is Paris.",
            tokens_prompt=8,
            tokens_generated=6,
            stop_reason="stop",
        )

        with patch("hfl.api.routes_openai._get_state", return_value=mock_state):
            with patch("hfl.api.routes_openai._ensure_model_loaded"):
                response = client.post(
                    "/v1/completions",
                    json={
                        "model": "test-model",
                        "prompt": "The capital of France is",
                        "max_tokens": 50,
                        "stream": False,
                    },
                )

        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "id" in data
        assert data["id"].startswith("cmpl-")
        assert data["object"] == "text_completion"
        assert "created" in data
        assert "model" in data

        # Choices
        assert "choices" in data
        choice = data["choices"][0]
        assert "text" in choice
        assert "index" in choice
        assert "finish_reason" in choice

        # Usage
        assert "usage" in data


class TestModelsEndpoint:
    """Tests for /v1/models endpoint."""

    def test_models_list_format(self, client):
        """Models list should match OpenAI format."""
        mock_manifest = MagicMock()
        mock_manifest.name = "test-model"
        mock_manifest.repo_id = "test-org/test-model"

        with patch("hfl.api.routes_openai.get_registry") as mock_get_registry:
            mock_get_registry.return_value.list_all.return_value = [mock_manifest]
            response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()

        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)

        if data["data"]:
            model = data["data"][0]
            assert "id" in model
            assert model["object"] == "model"
            assert "created" in model
            assert "owned_by" in model


class TestErrorFormats:
    """Tests for error response formats."""

    def test_model_not_found_error(self, client):
        """404 error should have proper format."""
        with patch("hfl.api.routes_openai._ensure_model_loaded") as mock_load:
            from fastapi import HTTPException
            mock_load.side_effect = HTTPException(404, "Model not found: nonexistent")

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "nonexistent",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert response.status_code == 404

    def test_validation_error(self, client):
        """Validation errors should have proper format."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "",  # Empty model name
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        # Should be 422 Unprocessable Entity
        assert response.status_code == 422


class TestRequestValidation:
    """Tests for request validation."""

    def test_temperature_bounds(self, client, mock_loaded_state):
        """Temperature should be clamped to valid range."""
        mock_state, mock_engine = mock_loaded_state
        mock_engine.chat.return_value = GenerationResult(
            text="Response",
            tokens_prompt=5,
            tokens_generated=1,
            stop_reason="stop",
        )

        with patch("hfl.api.routes_openai._get_state", return_value=mock_state):
            with patch("hfl.api.routes_openai._ensure_model_loaded"):
                # Temperature out of range should fail validation
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "temperature": 3.0,  # Above max of 2.0
                    },
                )

        # Should reject invalid temperature
        assert response.status_code == 422

    def test_max_tokens_bounds(self, client, mock_loaded_state):
        """Max tokens should be validated."""
        mock_state, mock_engine = mock_loaded_state

        with patch("hfl.api.routes_openai._get_state", return_value=mock_state):
            with patch("hfl.api.routes_openai._ensure_model_loaded"):
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": -1,  # Invalid
                    },
                )

        assert response.status_code == 422

    def test_empty_messages_rejected(self, client):
        """Empty messages array should be rejected."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [],  # Empty
            },
        )

        assert response.status_code == 422


class TestHeadersAndMetadata:
    """Tests for response headers and metadata."""

    def test_disclaimer_header_present(self, client, mock_loaded_state):
        """AI disclaimer header should be present on AI endpoints."""
        mock_state, mock_engine = mock_loaded_state
        mock_engine.chat.return_value = GenerationResult(
            text="Response",
            tokens_prompt=5,
            tokens_generated=1,
            stop_reason="stop",
        )

        with patch("hfl.api.routes_openai._get_state", return_value=mock_state):
            with patch("hfl.api.routes_openai._ensure_model_loaded"):
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

        # R9 compliance - disclaimer header
        assert "x-ai-disclaimer" in response.headers

    def test_content_type_json(self, client, mock_loaded_state):
        """Non-streaming responses should be JSON."""
        mock_state, mock_engine = mock_loaded_state
        mock_engine.chat.return_value = GenerationResult(
            text="Response",
            tokens_prompt=5,
            tokens_generated=1,
            stop_reason="stop",
        )

        with patch("hfl.api.routes_openai._get_state", return_value=mock_state):
            with patch("hfl.api.routes_openai._ensure_model_loaded"):
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": False,
                    },
                )

        assert "application/json" in response.headers["content-type"]
