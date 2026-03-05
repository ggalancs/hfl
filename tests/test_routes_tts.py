# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for TTS API routes."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hfl.engine.base import AudioResult


@pytest.fixture
def client(temp_config):
    """Test client for the API."""
    from hfl.api.server import app

    return TestClient(app)


@pytest.fixture
def client_with_tts_model(temp_config, sample_manifest):
    """Client with pre-loaded TTS model."""
    from hfl.api.server import app
    from hfl.api.state import get_state
    from hfl.models.registry import ModelRegistry

    state = get_state()

    # Create TTS manifest
    tts_manifest = sample_manifest
    tts_manifest.name = "bark-small"
    tts_manifest.repo_id = "suno/bark-small"

    # Register model
    registry = ModelRegistry()
    registry.add(tts_manifest)

    # Mock the TTS engine
    mock_engine = MagicMock()
    mock_engine.is_loaded = True
    mock_engine.model_name = "bark-small"
    mock_engine.supported_voices = ["v2/en_speaker_0", "v2/en_speaker_1"]
    mock_engine.supported_languages = ["en", "es"]
    mock_engine.synthesize.return_value = AudioResult(
        audio=b"RIFF" + b"\x00" * 100,  # Fake WAV header
        sample_rate=22050,
        duration=1.0,
        format="wav",
    )
    mock_engine.synthesize_stream.return_value = iter([b"chunk1", b"chunk2"])

    state.tts_engine = mock_engine
    state.current_tts_model = tts_manifest

    yield TestClient(app)

    # Cleanup
    state.tts_engine = None
    state.current_tts_model = None


class TestOpenAITTSEndpoint:
    """Tests for OpenAI-compatible TTS endpoint."""

    def test_speech_without_model_loaded(self, client):
        """Should return 404 when no TTS model is loaded."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "nonexistent",
                "input": "Hello world",
            },
        )

        assert response.status_code == 404

    def test_speech_with_loaded_model(self, client_with_tts_model):
        """Should synthesize audio when model is loaded."""
        with patch("hfl.api.routes_tts._ensure_tts_model_loaded"):
            response = client_with_tts_model.post(
                "/v1/audio/speech",
                json={
                    "model": "bark-small",
                    "input": "Hello world",
                    "voice": "alloy",
                    "response_format": "wav",
                },
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert "X-Audio-Duration" in response.headers
        assert "X-Audio-Sample-Rate" in response.headers
        assert response.content.startswith(b"RIFF")

    def test_speech_with_speed(self, client_with_tts_model):
        """Should accept speed parameter."""
        with patch("hfl.api.routes_tts._ensure_tts_model_loaded"):
            response = client_with_tts_model.post(
                "/v1/audio/speech",
                json={
                    "model": "bark-small",
                    "input": "Hello",
                    "speed": 1.5,
                },
            )

        assert response.status_code == 200

    def test_speech_input_too_long(self, client):
        """Should reject input over 4096 characters."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "bark-small",
                "input": "x" * 5000,
            },
        )

        assert response.status_code == 422  # Validation error

    def test_speech_invalid_speed(self, client):
        """Should reject invalid speed values."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "bark-small",
                "input": "Hello",
                "speed": 5.0,  # Max is 4.0
            },
        )

        assert response.status_code == 422


class TestNativeTTSEndpoint:
    """Tests for native HFL TTS endpoint."""

    def test_tts_without_model_loaded(self, client):
        """Should return 404 when no TTS model is loaded."""
        response = client.post(
            "/api/tts",
            json={
                "model": "nonexistent",
                "text": "Hello world",
            },
        )

        assert response.status_code == 404

    def test_tts_with_loaded_model(self, client_with_tts_model):
        """Should synthesize audio when model is loaded."""
        with patch("hfl.api.routes_tts._ensure_tts_model_loaded"):
            response = client_with_tts_model.post(
                "/api/tts",
                json={
                    "model": "bark-small",
                    "text": "Hello world",
                    "language": "en",
                    "format": "wav",
                },
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"

    def test_tts_with_custom_sample_rate(self, client_with_tts_model):
        """Should accept custom sample rate."""
        with patch("hfl.api.routes_tts._ensure_tts_model_loaded"):
            response = client_with_tts_model.post(
                "/api/tts",
                json={
                    "model": "bark-small",
                    "text": "Hello",
                    "sample_rate": 44100,
                },
            )

        assert response.status_code == 200

    def test_tts_streaming(self, client_with_tts_model):
        """Should support streaming mode."""
        with patch("hfl.api.routes_tts._ensure_tts_model_loaded"):
            response = client_with_tts_model.post(
                "/api/tts",
                json={
                    "model": "bark-small",
                    "text": "Hello",
                    "stream": True,
                },
            )

        assert response.status_code == 200
        # Check we got audio content (TestClient buffers streaming responses)
        assert response.headers.get("content-type") == "audio/wav"
        assert len(response.content) > 0


class TestTTSVoicesEndpoint:
    """Tests for TTS voices endpoint."""

    def test_voices_without_model(self, client):
        """Should return info message without model."""
        response = client.get("/api/tts/voices")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_voices_with_model(self, client_with_tts_model):
        """Should return voices for loaded model."""
        with patch("hfl.api.routes_tts._ensure_tts_model_loaded"):
            response = client_with_tts_model.get("/api/tts/voices?model=bark-small")

        assert response.status_code == 200
        data = response.json()
        assert "voices" in data
        assert "languages" in data
        assert isinstance(data["voices"], list)


class TestTTSModelsEndpoint:
    """Tests for TTS models listing endpoint."""

    def test_list_tts_models_empty(self, client):
        """Should return empty list when no TTS models."""
        response = client.get("/v1/audio/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)


class TestTTSSchemas:
    """Tests for TTS request/response schemas."""

    def test_openai_tts_request_defaults(self):
        """Should have correct defaults for OpenAI TTS request."""
        from hfl.api.routes_tts import OpenAITTSRequest

        req = OpenAITTSRequest(model="bark", input="Hello")

        assert req.voice == "alloy"
        assert req.response_format == "mp3"
        assert req.speed == 1.0

    def test_native_tts_request_defaults(self):
        """Should have correct defaults for native TTS request."""
        from hfl.api.routes_tts import NativeTTSRequest

        req = NativeTTSRequest(model="bark", text="Hello")

        assert req.voice == "default"
        assert req.language == "en"
        assert req.speed == 1.0
        assert req.sample_rate == 22050
        assert req.format == "wav"
        assert req.stream is False

    def test_format_to_content_type(self):
        """Should map formats to MIME types."""
        from hfl.api.routes_tts import _format_to_content_type

        assert _format_to_content_type("wav") == "audio/wav"
        assert _format_to_content_type("mp3") == "audio/mpeg"
        assert _format_to_content_type("ogg") == "audio/ogg"
        assert _format_to_content_type("unknown") == "audio/wav"  # Default

    def test_map_openai_format(self):
        """Should map OpenAI format names to internal formats."""
        from hfl.api.routes_tts import _map_openai_format

        assert _map_openai_format("mp3") == "mp3"
        assert _map_openai_format("wav") == "wav"
        assert _map_openai_format("opus") == "ogg"
        assert _map_openai_format("pcm") == "wav"
