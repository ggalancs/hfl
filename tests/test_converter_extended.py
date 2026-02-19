# SPDX-License-Identifier: HRUL-1.0
"""Extended tests for GGUF converter module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hfl.converter.gguf_converter import (
    UnsupportedModelError,
    check_model_convertibility,
    UNSUPPORTED_MODEL_TYPES,
    UNSUPPORTED_FILE_PATTERNS,
)


class TestCheckModelConvertibility:
    """Tests for check_model_convertibility function."""

    def test_no_config_json(self, tmp_path):
        """Test model directory without config.json."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        is_convertible, reason = check_model_convertibility(model_dir)

        assert is_convertible is False
        assert "config.json" in reason.lower()

    def test_unsupported_model_type(self, tmp_path):
        """Test with unsupported model type."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        config = {"model_type": "stable-diffusion"}
        (model_dir / "config.json").write_text(json.dumps(config))

        is_convertible, reason = check_model_convertibility(model_dir)

        assert is_convertible is False
        assert "stable-diffusion" in reason.lower()

    def test_lora_adapter(self, tmp_path):
        """Test with LoRA adapter model (no config.json, only adapter_config.json)."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # LoRA adapters typically don't have config.json, only adapter_config.json
        (model_dir / "adapter_model.safetensors").touch()
        (model_dir / "adapter_config.json").write_text("{}")

        is_convertible, reason = check_model_convertibility(model_dir)

        assert is_convertible is False
        assert "lora" in reason.lower() or "adapter" in reason.lower()

    def test_supported_model(self, tmp_path):
        """Test with supported model type."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        config = {"model_type": "llama"}
        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "model.safetensors").touch()

        is_convertible, reason = check_model_convertibility(model_dir)

        assert is_convertible is True
        assert reason == ""

    def test_qwen_model(self, tmp_path):
        """Test with Qwen model type."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        config = {"model_type": "qwen2"}
        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "model.safetensors").touch()

        is_convertible, reason = check_model_convertibility(model_dir)

        assert is_convertible is True

    def test_mistral_model(self, tmp_path):
        """Test with Mistral model type."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        config = {"model_type": "mistral"}
        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "model.safetensors").touch()

        is_convertible, reason = check_model_convertibility(model_dir)

        assert is_convertible is True


class TestUnsupportedModelTypes:
    """Tests for unsupported model type constants."""

    def test_contains_diffusion_models(self):
        """Test that diffusion models are in unsupported list."""
        assert "stable-diffusion" in UNSUPPORTED_MODEL_TYPES
        assert "flux" in UNSUPPORTED_MODEL_TYPES
        assert "sdxl" in UNSUPPORTED_MODEL_TYPES

    def test_contains_adapters(self):
        """Test that adapter types are in unsupported list."""
        assert "lora" in UNSUPPORTED_MODEL_TYPES
        assert "adapter" in UNSUPPORTED_MODEL_TYPES

    def test_contains_vision_models(self):
        """Test that vision models are in unsupported list."""
        assert "vit" in UNSUPPORTED_MODEL_TYPES
        assert "clip" in UNSUPPORTED_MODEL_TYPES


class TestUnsupportedFilePatterns:
    """Tests for unsupported file pattern constants."""

    def test_contains_adapter_files(self):
        """Test that adapter files are in unsupported patterns."""
        assert "adapter_model.safetensors" in UNSUPPORTED_FILE_PATTERNS
        assert "adapter_config.json" in UNSUPPORTED_FILE_PATTERNS

    def test_contains_diffusion_files(self):
        """Test that diffusion files are in unsupported patterns."""
        assert "diffusion_pytorch_model.safetensors" in UNSUPPORTED_FILE_PATTERNS


class TestGGUFConverterConstants:
    """Tests for GGUFConverter class constants."""

    def test_gguf_converter_can_be_imported(self):
        """Test that GGUFConverter can be imported."""
        from hfl.converter.gguf_converter import GGUFConverter

        assert GGUFConverter is not None
        # Just verify the class can be instantiated
        converter = GGUFConverter()
        assert hasattr(converter, "llama_cpp_dir")
        assert hasattr(converter, "convert_script")
        assert hasattr(converter, "quantize_bin")
