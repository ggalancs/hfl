# SPDX-License-Identifier: HRUL-1.0
"""Edge case tests for GGUF converter module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hfl.converter.gguf_converter import (
    check_model_convertibility,
    _get_llama_cpp_version,
)


class TestCheckModelConvertibilityEdgeCases:
    """Edge case tests for check_model_convertibility function."""

    def test_lora_adapter_via_config_content(self, tmp_path):
        """Test LoRA adapter detection via adapter_config in config.json."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Config with adapter_config key but no model_type
        config = {"adapter_config": {"r": 8, "lora_alpha": 16}}
        (model_dir / "config.json").write_text(json.dumps(config))

        is_convertible, reason = check_model_convertibility(model_dir)

        assert is_convertible is False
        assert "lora" in reason.lower()

    def test_lora_adapter_via_base_model_path(self, tmp_path):
        """Test LoRA adapter detection via base_model in config.json."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Config with base_model containing _name_or_path
        config = {"base_model": {"_name_or_path": "meta-llama/Llama-2-7b"}}
        (model_dir / "config.json").write_text(json.dumps(config))

        is_convertible, reason = check_model_convertibility(model_dir)

        assert is_convertible is False
        assert "lora" in reason.lower()

    def test_no_model_type_generic_error(self, tmp_path):
        """Test error when config has no model_type and no adapter indicators."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Config without model_type or adapter indicators
        config = {"some_random_key": "value"}
        (model_dir / "config.json").write_text(json.dumps(config))

        is_convertible, reason = check_model_convertibility(model_dir)

        assert is_convertible is False
        assert "model_type" in reason.lower()

    def test_invalid_json_config(self, tmp_path):
        """Test handling of invalid JSON in config.json."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Invalid JSON
        (model_dir / "config.json").write_text("{ invalid json }")

        is_convertible, reason = check_model_convertibility(model_dir)

        assert is_convertible is False
        assert "could not read" in reason.lower()

    def test_config_read_error(self, tmp_path):
        """Test handling of file read errors."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Create config.json as a directory (causes read error)
        config_path = model_dir / "config.json"
        config_path.mkdir()

        is_convertible, reason = check_model_convertibility(model_dir)

        assert is_convertible is False

    def test_diffusion_model_via_file_pattern(self, tmp_path):
        """Test diffusion model detection via file patterns."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # No config.json but has diffusion file
        (model_dir / "diffusion_pytorch_model.safetensors").touch()

        is_convertible, reason = check_model_convertibility(model_dir)

        assert is_convertible is False
        assert "diffusion" in reason.lower() or "image" in reason.lower()


class TestGetLlamaCppVersion:
    """Tests for _get_llama_cpp_version function."""

    def test_get_version_success(self, tmp_path):
        """Test getting version when git is available."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "abc1234"
            mock_run.return_value = mock_result

            result = _get_llama_cpp_version(tmp_path)

            assert result == "abc1234"

    def test_get_version_git_failure(self, tmp_path):
        """Test getting version when git fails."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_run.return_value = mock_result

            result = _get_llama_cpp_version(tmp_path)

            assert result == "unknown"

    def test_get_version_exception(self, tmp_path):
        """Test getting version when exception occurs."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Git not found")

            result = _get_llama_cpp_version(tmp_path)

            assert result == "unknown"


class TestGGUFConverterEnsureTools:
    """Tests for GGUFConverter.ensure_tools method."""

    def test_tools_already_exist(self, tmp_path):
        """Test when tools already exist, no setup needed."""
        from hfl.converter.gguf_converter import GGUFConverter

        converter = GGUFConverter()

        # Mock the paths to exist
        with patch.object(converter, 'llama_cpp_dir', tmp_path):
            convert_script = tmp_path / "convert_hf_to_gguf.py"
            convert_script.touch()

            quantize_dir = tmp_path / "build" / "bin"
            quantize_dir.mkdir(parents=True)
            quantize_bin = quantize_dir / "llama-quantize"
            quantize_bin.touch()

            converter.convert_script = convert_script
            converter.quantize_bin = quantize_bin

            # Should not raise and should return early
            converter.ensure_tools()
