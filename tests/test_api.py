# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests para el módulo API (server, routes_openai, routes_native)."""

import pytest
import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client(temp_config):
    """Cliente de test para la API."""
    from hfl.api.server import app
    return TestClient(app)


@pytest.fixture
def client_with_model(temp_config, sample_manifest):
    """Cliente con modelo pre-cargado."""
    from hfl.api.server import app, state
    from hfl.models.registry import ModelRegistry

    # Registrar modelo
    registry = ModelRegistry()
    registry.add(sample_manifest)

    # Mock del engine
    mock_engine = MagicMock()
    mock_engine.is_loaded = True
    mock_engine.chat.return_value = MagicMock(
        text="Hello!",
        tokens_prompt=10,
        tokens_generated=5,
        stop_reason="stop",
    )
    mock_engine.generate.return_value = MagicMock(
        text="Generated",
        tokens_prompt=5,
        tokens_generated=10,
        stop_reason="stop",
    )
    mock_engine.chat_stream.return_value = iter(["Hello", " world", "!"])
    mock_engine.generate_stream.return_value = iter(["Gen", "era", "ted"])

    state.engine = mock_engine
    state.current_model = sample_manifest

    yield TestClient(app)

    # Cleanup
    state.engine = None
    state.current_model = None


class TestRootEndpoints:
    """Tests para endpoints raíz."""

    def test_root(self, client):
        """Verifica endpoint raíz."""
        response = client.get("/")

        assert response.status_code == 200
        assert response.json()["status"] == "hfl is running"

    def test_health_no_model(self, client):
        """Verifica health sin modelo cargado."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is False
        assert data["current_model"] is None

    def test_health_with_model(self, client_with_model):
        """Verifica health con modelo cargado."""
        response = client_with_model.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True
        assert data["current_model"] == "test-model-q4_k_m"


class TestOpenAIEndpoints:
    """Tests para endpoints compatibles con OpenAI."""

    def test_list_models_empty(self, client):
        """Lista modelos vacía."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)

    def test_list_models_with_data(self, client, temp_config, sample_manifest):
        """Lista modelos con datos."""
        from hfl.models.registry import ModelRegistry

        registry = ModelRegistry()
        registry.add(sample_manifest)

        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model-q4_k_m"
        assert data["data"][0]["object"] == "model"

    def test_chat_completions_no_model(self, client):
        """Chat completions sin modelo."""
        response = client.post("/v1/chat/completions", json={
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "hi"}],
        })

        assert response.status_code == 404

    def test_chat_completions_success(self, client_with_model):
        """Chat completions exitoso."""
        response = client_with_model.post("/v1/chat/completions", json={
            "model": "test-model-q4_k_m",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        })

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert "usage" in data

    def test_chat_completions_with_system(self, client_with_model):
        """Chat completions con mensaje de sistema."""
        response = client_with_model.post("/v1/chat/completions", json={
            "model": "test-model-q4_k_m",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ],
        })

        assert response.status_code == 200

    def test_chat_completions_stream(self, client_with_model):
        """Chat completions con streaming."""
        response = client_with_model.post("/v1/chat/completions", json={
            "model": "test-model-q4_k_m",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Verificar formato SSE
        content = response.text
        assert "data:" in content
        assert "[DONE]" in content

    def test_completions_no_model(self, client):
        """Completions sin modelo."""
        response = client.post("/v1/completions", json={
            "model": "nonexistent",
            "prompt": "Hello",
        })

        assert response.status_code == 404

    def test_completions_success(self, client_with_model):
        """Completions exitoso."""
        response = client_with_model.post("/v1/completions", json={
            "model": "test-model-q4_k_m",
            "prompt": "Once upon a time",
            "max_tokens": 50,
            "stream": False,
        })

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["object"] == "text_completion"
        assert data["choices"][0]["text"] == "Generated"

    def test_completions_with_list_prompt(self, client_with_model):
        """Completions con prompt como lista."""
        response = client_with_model.post("/v1/completions", json={
            "model": "test-model-q4_k_m",
            "prompt": ["First prompt", "Second prompt"],
            "max_tokens": 50,
        })

        assert response.status_code == 200

    def test_completions_stream(self, client_with_model):
        """Completions con streaming."""
        response = client_with_model.post("/v1/completions", json={
            "model": "test-model-q4_k_m",
            "prompt": "Hello",
            "stream": True,
        })

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    def test_chat_completions_params(self, client_with_model):
        """Verifica que se pasan los parámetros correctamente."""
        response = client_with_model.post("/v1/chat/completions", json={
            "model": "test-model-q4_k_m",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.5,
            "top_p": 0.8,
            "max_tokens": 100,
            "stop": ["END"],
            "seed": 42,
        })

        assert response.status_code == 200


class TestNativeEndpoints:
    """Tests para endpoints nativos (Ollama-compatible)."""

    def test_api_tags_empty(self, client):
        """Lista tags vacía."""
        response = client.get("/api/tags")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_api_tags_with_models(self, client, temp_config, sample_manifest):
        """Lista tags con modelos."""
        from hfl.models.registry import ModelRegistry

        registry = ModelRegistry()
        registry.add(sample_manifest)

        response = client.get("/api/tags")

        assert response.status_code == 200
        data = response.json()
        assert len(data["models"]) == 1
        model = data["models"][0]
        assert model["name"] == "test-model-q4_k_m"
        assert "details" in model
        assert model["details"]["format"] == "gguf"

    def test_api_version(self, client):
        """Verifica endpoint de versión."""
        response = client.get("/api/version")

        assert response.status_code == 200
        assert "version" in response.json()

    def test_head_root(self, client):
        """Verifica HEAD request."""
        response = client.head("/")

        assert response.status_code == 200

    def test_api_generate_no_model(self, client):
        """Generate sin modelo."""
        response = client.post("/api/generate", json={
            "model": "nonexistent",
            "prompt": "Hello",
        })

        assert response.status_code == 404

    def test_api_generate_success(self, client_with_model):
        """Generate exitoso."""
        response = client_with_model.post("/api/generate", json={
            "model": "test-model-q4_k_m",
            "prompt": "Hello",
            "stream": False,
        })

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test-model-q4_k_m"
        assert data["done"] is True
        assert "response" in data

    def test_api_generate_stream(self, client_with_model):
        """Generate con streaming."""
        response = client_with_model.post("/api/generate", json={
            "model": "test-model-q4_k_m",
            "prompt": "Hello",
            "stream": True,
        })

        assert response.status_code == 200
        assert "application/x-ndjson" in response.headers["content-type"]

        # Verificar formato NDJSON
        lines = [l for l in response.text.strip().split("\n") if l]
        for line in lines:
            data = json.loads(line)
            assert "model" in data
            assert "done" in data

    def test_api_chat_no_model(self, client):
        """Chat sin modelo."""
        response = client.post("/api/chat", json={
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "hi"}],
        })

        assert response.status_code == 404

    def test_api_chat_success(self, client_with_model):
        """Chat exitoso."""
        response = client_with_model.post("/api/chat", json={
            "model": "test-model-q4_k_m",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        })

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test-model-q4_k_m"
        assert data["done"] is True
        assert "message" in data
        assert data["message"]["role"] == "assistant"

    def test_api_chat_stream(self, client_with_model):
        """Chat con streaming."""
        response = client_with_model.post("/api/chat", json={
            "model": "test-model-q4_k_m",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })

        assert response.status_code == 200
        assert "application/x-ndjson" in response.headers["content-type"]

    def test_api_generate_with_options(self, client_with_model):
        """Generate con opciones."""
        response = client_with_model.post("/api/generate", json={
            "model": "test-model-q4_k_m",
            "prompt": "Hello",
            "stream": False,
            "options": {
                "temperature": 0.5,
                "top_p": 0.8,
                "top_k": 50,
                "num_predict": 100,
                "repeat_penalty": 1.2,
                "seed": 42,
            },
        })

        assert response.status_code == 200


class TestMiddleware:
    """Tests para middleware."""

    def test_cors_headers(self, client):
        """Verifica headers CORS."""
        response = client.options("/", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        })

        # CORS middleware debería responder
        assert response.status_code in [200, 400]

    def test_error_handling(self, client):
        """Verifica manejo de errores."""
        response = client.post("/v1/chat/completions", json={
            "model": "invalid",
            "messages": "invalid",  # Debería ser lista
        })

        assert response.status_code in [404, 422]  # Not found o validation error


class TestServerState:
    """Tests para el estado del servidor."""

    def test_initial_state(self):
        """Verifica estado inicial."""
        from hfl.api.server import ServerState

        state = ServerState()

        assert state.engine is None
        assert state.current_model is None

    def test_start_server_function_exists(self):
        """Verifica que start_server existe."""
        from hfl.api.server import start_server

        assert callable(start_server)
