# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Global pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def temp_dir():
    """Creates a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config(temp_dir, monkeypatch):
    """Creates an isolated temporary configuration for tests."""
    # Import after the module is available
    from hfl.config import HFLConfig

    test_config = HFLConfig(home_dir=temp_dir)
    test_config.ensure_dirs()

    # Monkeypatch the global configuration
    import hfl.config
    monkeypatch.setattr(hfl.config, "config", test_config)

    # We also need to patch where it's imported directly
    import hfl.models.registry
    monkeypatch.setattr(hfl.models.registry, "config", test_config)

    import hfl.converter.gguf_converter
    monkeypatch.setattr(hfl.converter.gguf_converter, "config", test_config)

    import hfl.hub.downloader
    monkeypatch.setattr(hfl.hub.downloader, "config", test_config)

    yield test_config


@pytest.fixture
def mock_hf_api():
    """Mock of the HuggingFace API."""
    with patch("huggingface_hub.HfApi") as mock:
        api_instance = MagicMock()
        mock.return_value = api_instance
        yield api_instance


@pytest.fixture
def sample_model_info():
    """Sample model information."""
    mock_info = MagicMock()
    mock_info.id = "test-org/test-model"
    mock_info.siblings = [
        MagicMock(rfilename="model.safetensors"),
        MagicMock(rfilename="config.json"),
        MagicMock(rfilename="tokenizer.json"),
    ]
    return mock_info


@pytest.fixture
def sample_gguf_model_info():
    """Sample GGUF model information."""
    mock_info = MagicMock()
    mock_info.id = "test-org/test-model-gguf"
    mock_info.siblings = [
        MagicMock(rfilename="model-Q4_K_M.gguf"),
        MagicMock(rfilename="model-Q5_K_M.gguf"),
        MagicMock(rfilename="config.json"),
    ]
    return mock_info


@pytest.fixture
def sample_manifest():
    """Sample model manifest."""
    from hfl.models.manifest import ModelManifest
    return ModelManifest(
        name="test-model-q4_k_m",
        repo_id="test-org/test-model",
        local_path="/tmp/test-model",
        format="gguf",
        size_bytes=5 * 1024**3,  # 5 GB
        quantization="Q4_K_M",
        architecture="llama",
        parameters="7B",
        context_length=4096,
    )


@pytest.fixture
def mock_llama_model():
    """Mock of llama-cpp model."""
    mock = MagicMock()
    mock.return_value = {
        "choices": [{"text": "Hello, world!", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    mock.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    return mock


@pytest.fixture
def populated_registry(temp_config, sample_manifest):
    """Registry with sample models."""
    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    registry.add(sample_manifest)

    # Add another model
    from hfl.models.manifest import ModelManifest
    registry.add(ModelManifest(
        name="another-model-q5_k_m",
        repo_id="other-org/another-model",
        local_path="/tmp/another-model",
        format="gguf",
        size_bytes=10 * 1024**3,
        quantization="Q5_K_M",
    ))

    return registry


@pytest.fixture
def mock_llama_cpp():
    """Mock of the complete llama_cpp module."""
    mock_llama = MagicMock()
    mock_llama_class = MagicMock()
    mock_llama.Llama = mock_llama_class

    with patch.dict(sys.modules, {"llama_cpp": mock_llama}):
        yield mock_llama_class
