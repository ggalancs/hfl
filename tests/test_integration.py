# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""End-to-end integration tests."""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import json


@pytest.fixture(autouse=True)
def mock_llama_cpp():
    """Mock llama_cpp for all integration tests."""
    mock = MagicMock()
    mock.Llama = MagicMock()
    with patch.dict(sys.modules, {"llama_cpp": mock}):
        yield mock


class TestFullWorkflow:
    """Tests for complete workflow."""

    def test_pull_list_inspect_rm_workflow(self, temp_config):
        """Complete workflow: pull -> list -> inspect -> rm."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest
        from hfl.hub.resolver import ResolvedModel

        # Simulate pull by creating manifest directly
        manifest = ModelManifest(
            name="workflow-test-q4_k_m",
            repo_id="test/workflow-model",
            local_path=str(temp_config.models_dir / "workflow-model.gguf"),
            format="gguf",
            size_bytes=100 * 1024**2,
            quantization="Q4_K_M",
        )

        # Create file
        (temp_config.models_dir / "workflow-model.gguf").write_bytes(b"GGUF")

        # Register
        registry = ModelRegistry()
        registry.add(manifest)

        # List
        models = registry.list_all()
        assert len(models) == 1
        assert models[0].name == "workflow-test-q4_k_m"

        # Inspect
        model = registry.get("workflow-test-q4_k_m")
        assert model is not None
        assert model.quantization == "Q4_K_M"
        assert model.format == "gguf"

        # Remove
        result = registry.remove("workflow-test-q4_k_m")
        assert result is True
        assert registry.get("workflow-test-q4_k_m") is None

    def test_resolve_and_detect_format(self, temp_config):
        """Model resolution and format detection."""
        from hfl.converter.formats import detect_format, ModelFormat

        # Create model structures
        gguf_model = temp_config.models_dir / "gguf-model"
        gguf_model.mkdir()
        (gguf_model / "model.gguf").write_bytes(b"GGUF")

        st_model = temp_config.models_dir / "st-model"
        st_model.mkdir()
        (st_model / "model.safetensors").write_bytes(b"ST")
        (st_model / "config.json").write_text("{}")

        # Detect formats
        assert detect_format(gguf_model) == ModelFormat.GGUF
        assert detect_format(st_model) == ModelFormat.SAFETENSORS

    def test_engine_selection_workflow(self, temp_config):
        """Engine selection based on format."""
        from hfl.engine.selector import select_engine
        from hfl.engine.llama_cpp import LlamaCppEngine

        # Create GGUF model
        gguf_model = temp_config.models_dir / "test.gguf"
        gguf_model.write_bytes(b"GGUF")

        # Select engine
        engine = select_engine(gguf_model)

        assert isinstance(engine, LlamaCppEngine)

    def test_registry_persistence_workflow(self, temp_config):
        """Registry persistence between sessions."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        # Session 1: Add models
        registry1 = ModelRegistry()
        for i in range(3):
            manifest = ModelManifest(
                name=f"persist-model-{i}",
                repo_id=f"org/persist-model-{i}",
                local_path=f"/path/{i}",
                format="gguf",
                size_bytes=i * 1024**3,
            )
            registry1.add(manifest)

        # Session 2: Load and verify
        registry2 = ModelRegistry()
        models = registry2.list_all()

        assert len(models) == 3
        for i in range(3):
            model = registry2.get(f"persist-model-{i}")
            assert model is not None
            assert model.size_bytes == i * 1024**3


class TestAPIIntegration:
    """API integration tests."""

    def test_api_model_lifecycle(self, temp_config, sample_manifest):
        """Model lifecycle through the API."""
        from fastapi.testclient import TestClient
        from hfl.api.server import app, state
        from hfl.models.registry import ModelRegistry

        client = TestClient(app)

        # Initially no models
        response = client.get("/v1/models")
        assert response.status_code == 200
        assert len(response.json()["data"]) == 0

        # Register model
        registry = ModelRegistry()
        registry.add(sample_manifest)

        # Now there is a model
        response = client.get("/v1/models")
        assert response.status_code == 200
        assert len(response.json()["data"]) == 1

        # Also visible in Ollama API
        response = client.get("/api/tags")
        assert response.status_code == 200
        assert len(response.json()["models"]) == 1

        # Cleanup
        registry.remove(sample_manifest.name)

    def test_openai_ollama_compatibility(self, temp_config, sample_manifest):
        """Compatibility between OpenAI and Ollama endpoints."""
        from fastapi.testclient import TestClient
        from hfl.api.server import app, state
        from hfl.models.registry import ModelRegistry

        client = TestClient(app)

        # Register model
        registry = ModelRegistry()
        registry.add(sample_manifest)

        # Mock engine
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.chat.return_value = MagicMock(
            text="Test response",
            tokens_prompt=10,
            tokens_generated=5,
            stop_reason="stop",
        )
        state.engine = mock_engine
        state.current_model = sample_manifest

        try:
            # OpenAI endpoint
            openai_response = client.post("/v1/chat/completions", json={
                "model": sample_manifest.name,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            })

            # Ollama endpoint
            ollama_response = client.post("/api/chat", json={
                "model": sample_manifest.name,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            })

            assert openai_response.status_code == 200
            assert ollama_response.status_code == 200

            # Both return the same content
            openai_content = openai_response.json()["choices"][0]["message"]["content"]
            ollama_content = ollama_response.json()["message"]["content"]
            assert openai_content == ollama_content == "Test response"

        finally:
            state.engine = None
            state.current_model = None
            registry.remove(sample_manifest.name)


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_quantization_level(self):
        """Handling of invalid quantization level."""
        from hfl.hub.resolver import _detect_quant

        # Should not fail, just return None
        result = _detect_quant("model-INVALID.gguf")
        assert result is None

    def test_corrupted_manifest(self, temp_config):
        """Handling of corrupted manifest in registry."""
        from hfl.models.registry import ModelRegistry

        # Write corrupted data
        temp_config.registry_path.write_text('[{"name": "incomplete"}]')

        # Should handle gracefully
        try:
            registry = ModelRegistry()
            # May or may not load depending on validation
        except Exception:
            # Acceptable if it fails in a controlled manner
            pass

    def test_missing_model_path(self, temp_config):
        """Handling of non-existent model path."""
        from hfl.converter.formats import detect_format, ModelFormat

        result = detect_format(Path("/nonexistent/model/path"))
        assert result == ModelFormat.UNKNOWN


class TestPerformance:
    """Basic performance tests."""

    def test_registry_many_models(self, temp_config):
        """Registry with many models."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        registry = ModelRegistry()

        # Add 100 models
        for i in range(100):
            manifest = ModelManifest(
                name=f"perf-model-{i:03d}",
                repo_id=f"org/perf-model-{i}",
                local_path=f"/path/{i}",
                format="gguf",
            )
            registry.add(manifest)

        # Verify loading
        models = registry.list_all()
        assert len(models) == 100

        # Fast search
        model = registry.get("perf-model-050")
        assert model is not None

    def test_format_detection_many_files(self, temp_dir):
        """Format detection with many files."""
        from hfl.converter.formats import detect_format, ModelFormat

        # Create many files
        for i in range(50):
            (temp_dir / f"file_{i}.txt").write_text(f"content {i}")

        # Add GGUF file at the end
        (temp_dir / "model.gguf").write_bytes(b"GGUF")

        # Should find it
        result = detect_format(temp_dir)
        assert result == ModelFormat.GGUF
