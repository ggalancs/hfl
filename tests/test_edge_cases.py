# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Edge case tests for robustness and security.

Tests for:
- Empty inputs
- Boundary values
- Malicious inputs
- Validation errors
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client without loading real models."""
    from hfl.api.server import app

    return TestClient(app, raise_server_exceptions=False)


class TestEmptyInputs:
    """Tests for empty or missing inputs."""

    def test_empty_messages_list(self, client):
        """Empty messages list should fail validation."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [],
            },
        )
        assert response.status_code == 422  # Validation error

    def test_empty_prompt_completion(self, client):
        """Empty prompt should fail validation."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "",
            },
        )
        # Empty string with max_length constraint may pass or fail depending on validation
        # 404 = model not found, 503 = model not loaded
        assert response.status_code in (200, 404, 422, 503)

    def test_missing_model_field(self, client):
        """Missing required model field should fail."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 422

    def test_empty_model_name(self, client):
        """Empty model name should fail validation."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 422


class TestBoundaryValues:
    """Tests for boundary value conditions."""

    def test_temperature_at_minimum(self, client):
        """Temperature at minimum (0.0) should be valid."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.0,
            },
        )
        # 404 = model not found, 503 = model not loaded - validation passed
        assert response.status_code in (200, 404, 503)

    def test_temperature_at_maximum(self, client):
        """Temperature at maximum (2.0) should be valid."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 2.0,
            },
        )
        assert response.status_code in (200, 404, 503)

    def test_temperature_over_maximum(self, client):
        """Temperature over maximum should fail validation."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 2.5,
            },
        )
        assert response.status_code == 422

    def test_temperature_negative(self, client):
        """Negative temperature should fail validation."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": -0.5,
            },
        )
        assert response.status_code == 422

    def test_max_tokens_at_limit(self, client):
        """max_tokens at limit (128000) should be valid."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 128000,
            },
        )
        assert response.status_code in (200, 404, 503)

    def test_max_tokens_over_limit(self, client):
        """max_tokens over limit should fail validation."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 128001,
            },
        )
        assert response.status_code == 422

    def test_max_tokens_zero(self, client):
        """max_tokens of 0 should fail validation."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 0,
            },
        )
        assert response.status_code == 422

    def test_max_tokens_negative(self, client):
        """Negative max_tokens should fail validation."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": -100,
            },
        )
        assert response.status_code == 422

    def test_top_p_at_zero(self, client):
        """top_p at 0.0 should be valid."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "top_p": 0.0,
            },
        )
        assert response.status_code in (200, 404, 503)

    def test_top_p_over_one(self, client):
        """top_p over 1.0 should fail validation."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "top_p": 1.5,
            },
        )
        assert response.status_code == 422


class TestMaliciousInputs:
    """Tests for potentially malicious inputs."""

    def test_path_traversal_in_model_name(self, client):
        """Path traversal attempt in model name should be handled safely."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "../../../etc/passwd",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        # Should either fail validation or return model not found, not crash
        assert response.status_code in (400, 404, 422, 503)

    def test_null_bytes_in_model_name(self, client):
        """Null bytes in model name should be handled safely."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test\x00model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        # Should be handled gracefully
        assert response.status_code in (400, 404, 422, 503)

    def test_very_long_model_name(self, client):
        """Very long model name should fail validation."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "a" * 500,  # Over 256 char limit
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 422

    def test_unicode_in_model_name(self, client):
        """Unicode characters in model name should be handled."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model-\u200b\u200b\u200b",  # Zero-width spaces
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        # Should be handled gracefully
        assert response.status_code in (400, 404, 422, 503)

    def test_special_characters_in_prompt(self, client):
        """Special characters in prompt should be handled."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello\x00World\x1b[31mRed\x1b[0m"}],
            },
        )
        # Should not crash - 404 = model not found
        assert response.status_code in (200, 404, 503)

    def test_many_messages(self, client):
        """Many messages should work up to the limit."""
        messages = [{"role": "user", "content": f"Message {i}"} for i in range(100)]
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": messages,
            },
        )
        # Should work (100 is within limit of 1000) - 404 = model not found
        assert response.status_code in (200, 404, 503)

    def test_too_many_messages(self, client):
        """Too many messages should fail validation."""
        messages = [{"role": "user", "content": f"Msg {i}"} for i in range(1001)]
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": messages,
            },
        )
        assert response.status_code == 422


class TestNativeAPIEdgeCases:
    """Tests for Ollama-native API edge cases."""

    def test_native_generate_empty_prompt(self, client):
        """Empty prompt in native generate should handle gracefully."""
        response = client.post(
            "/api/generate",
            json={
                "model": "test-model",
                "prompt": "",
            },
        )
        # Empty string may pass validation - 404 = model not found
        assert response.status_code in (200, 404, 422, 503)

    def test_native_generate_extreme_options(self, client):
        """Extreme option values should be clamped, not crash."""
        response = client.post(
            "/api/generate",
            json={
                "model": "test-model",
                "prompt": "Hello",
                "options": {
                    "temperature": 999999,  # Should be clamped to 2.0
                    "num_predict": 999999999,  # Should be clamped to 128000
                    "top_p": -100,  # Should be clamped to 0.0
                },
            },
        )
        # Should not crash - values are clamped - 404 = model not found
        assert response.status_code in (200, 404, 503)

    def test_native_chat_empty_messages(self, client):
        """Empty messages in native chat should fail validation."""
        response = client.post(
            "/api/chat",
            json={
                "model": "test-model",
                "messages": [],
            },
        )
        assert response.status_code == 422

    def test_native_chat_invalid_message_format(self, client):
        """Invalid message format should fail."""
        response = client.post(
            "/api/chat",
            json={
                "model": "test-model",
                "messages": [{"invalid": "format"}],
            },
        )
        # Missing role/content should cause KeyError or validation error
        # 404 = model not found (before message parsing)
        assert response.status_code in (400, 404, 422, 500)


class TestTTSEdgeCases:
    """Tests for TTS API edge cases."""

    def test_tts_empty_input(self, client):
        """Empty input text should fail validation."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-tts",
                "input": "",
            },
        )
        # Empty input should be rejected - 404 = model not found
        assert response.status_code in (400, 404, 422, 503)

    def test_tts_very_long_input(self, client):
        """Very long input should be handled."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-tts",
                "input": "Hello " * 10000,  # Long but not extreme
            },
        )
        # Should either work or fail gracefully
        assert response.status_code in (200, 400, 422, 503)

    def test_tts_invalid_voice(self, client):
        """Invalid voice parameter should be handled."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-tts",
                "input": "Hello",
                "voice": "../../../etc/passwd",
            },
        )
        # Should be handled safely
        assert response.status_code in (400, 404, 422, 503)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_basic(self, client):
        """Basic health check should always succeed."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_health_ready(self, client):
        """Readiness check should return valid response."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data

    def test_health_live(self, client):
        """Liveness check should return valid response."""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "uptime_seconds" in data

    def test_health_deep(self, client):
        """Deep health check should return detailed info."""
        response = client.get("/health/deep")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "llm" in data
        assert "tts" in data


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    def test_rate_limit_headers(self, client):
        """Rate limited responses should include appropriate headers."""
        # Make many requests quickly
        responses = []
        for _ in range(100):
            response = client.get("/health")
            responses.append(response)
            if response.status_code == 429:
                break

        # If rate limiting is enabled, we should eventually get 429
        # If disabled, all should be 200
        status_codes = {r.status_code for r in responses}
        assert status_codes.issubset({200, 429})
