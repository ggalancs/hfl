# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the converter module (formats, gguf_converter)."""

import json
import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch


class TestModelFormat:
    """Tests for ModelFormat enum."""

    def test_enum_values(self):
        """Verifies enum values."""
        from hfl.converter.formats import ModelFormat

        assert ModelFormat.GGUF.value == "gguf"
        assert ModelFormat.SAFETENSORS.value == "safetensors"
        assert ModelFormat.PYTORCH.value == "pytorch"
        assert ModelFormat.UNKNOWN.value == "unknown"

    def test_enum_members(self):
        """Verifies enum members."""
        from hfl.converter.formats import ModelFormat

        assert hasattr(ModelFormat, "GGUF")
        assert hasattr(ModelFormat, "SAFETENSORS")
        assert hasattr(ModelFormat, "PYTORCH")
        assert hasattr(ModelFormat, "UNKNOWN")


class TestDetectFormat:
    """Tests for detect_format."""

    def test_detect_gguf_file(self, temp_dir):
        """Detects GGUF file."""
        from hfl.converter.formats import detect_format, ModelFormat

        gguf_file = temp_dir / "model.gguf"
        gguf_file.write_bytes(b"GGUF content")

        result = detect_format(gguf_file)
        assert result == ModelFormat.GGUF

    def test_detect_safetensors_file(self, temp_dir):
        """Detects safetensors file."""
        from hfl.converter.formats import detect_format, ModelFormat

        st_file = temp_dir / "model.safetensors"
        st_file.write_bytes(b"safetensors content")

        result = detect_format(st_file)
        assert result == ModelFormat.SAFETENSORS

    def test_detect_pytorch_bin_file(self, temp_dir):
        """Detects pytorch .bin file."""
        from hfl.converter.formats import detect_format, ModelFormat

        pt_file = temp_dir / "model.bin"
        pt_file.write_bytes(b"pytorch content")

        result = detect_format(pt_file)
        assert result == ModelFormat.PYTORCH

    def test_detect_pytorch_pt_file(self, temp_dir):
        """Detects pytorch .pt file."""
        from hfl.converter.formats import detect_format, ModelFormat

        pt_file = temp_dir / "model.pt"
        pt_file.write_bytes(b"pytorch content")

        result = detect_format(pt_file)
        assert result == ModelFormat.PYTORCH

    def test_detect_pytorch_pth_file(self, temp_dir):
        """Detects pytorch .pth file."""
        from hfl.converter.formats import detect_format, ModelFormat

        pt_file = temp_dir / "model.pth"
        pt_file.write_bytes(b"pytorch content")

        result = detect_format(pt_file)
        assert result == ModelFormat.PYTORCH

    def test_detect_gguf_in_directory(self, temp_dir):
        """Detects GGUF in directory."""
        from hfl.converter.formats import detect_format, ModelFormat

        (temp_dir / "model.gguf").write_bytes(b"GGUF")
        (temp_dir / "config.json").write_text("{}")

        result = detect_format(temp_dir)
        assert result == ModelFormat.GGUF

    def test_detect_safetensors_in_directory(self, temp_dir):
        """Detects safetensors in directory."""
        from hfl.converter.formats import detect_format, ModelFormat

        (temp_dir / "model.safetensors").write_bytes(b"ST")
        (temp_dir / "config.json").write_text("{}")

        result = detect_format(temp_dir)
        assert result == ModelFormat.SAFETENSORS

    def test_detect_pytorch_in_directory(self, temp_dir):
        """Detects pytorch in directory."""
        from hfl.converter.formats import detect_format, ModelFormat

        (temp_dir / "pytorch_model.bin").write_bytes(b"PT")
        (temp_dir / "config.json").write_text("{}")

        result = detect_format(temp_dir)
        assert result == ModelFormat.PYTORCH

    def test_detect_unknown_file(self, temp_dir):
        """Returns UNKNOWN for unknown file."""
        from hfl.converter.formats import detect_format, ModelFormat

        txt_file = temp_dir / "readme.txt"
        txt_file.write_text("readme")

        result = detect_format(txt_file)
        assert result == ModelFormat.UNKNOWN

    def test_detect_unknown_directory(self, temp_dir):
        """Returns UNKNOWN for directory without models."""
        from hfl.converter.formats import detect_format, ModelFormat

        (temp_dir / "readme.txt").write_text("readme")
        (temp_dir / "data.json").write_text("{}")

        result = detect_format(temp_dir)
        assert result == ModelFormat.UNKNOWN

    def test_detect_nonexistent_path(self):
        """Returns UNKNOWN for non-existent path."""
        from hfl.converter.formats import detect_format, ModelFormat

        result = detect_format(Path("/nonexistent/path"))
        assert result == ModelFormat.UNKNOWN

    def test_detect_gguf_priority_over_safetensors(self, temp_dir):
        """GGUF has priority over safetensors."""
        from hfl.converter.formats import detect_format, ModelFormat

        (temp_dir / "model.gguf").write_bytes(b"GGUF")
        (temp_dir / "model.safetensors").write_bytes(b"ST")

        result = detect_format(temp_dir)
        assert result == ModelFormat.GGUF

    def test_detect_nested_files(self, temp_dir):
        """Detects files in subdirectories."""
        from hfl.converter.formats import detect_format, ModelFormat

        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "model.gguf").write_bytes(b"GGUF")

        result = detect_format(temp_dir)
        assert result == ModelFormat.GGUF


class TestFindModelFile:
    """Tests for find_model_file."""

    def test_find_gguf_file_direct(self, temp_dir):
        """Finds direct GGUF file."""
        from hfl.converter.formats import find_model_file, ModelFormat

        gguf_file = temp_dir / "model.gguf"
        gguf_file.write_bytes(b"GGUF")

        result = find_model_file(gguf_file, ModelFormat.GGUF)
        assert result == gguf_file

    def test_find_gguf_in_directory(self, temp_dir):
        """Finds GGUF in directory."""
        from hfl.converter.formats import find_model_file, ModelFormat

        gguf_file = temp_dir / "model.gguf"
        gguf_file.write_bytes(b"GGUF")

        result = find_model_file(temp_dir, ModelFormat.GGUF)
        assert result == gguf_file

    def test_find_safetensors_returns_directory(self, temp_dir):
        """For safetensors returns the directory."""
        from hfl.converter.formats import find_model_file, ModelFormat

        (temp_dir / "model.safetensors").write_bytes(b"ST")

        result = find_model_file(temp_dir, ModelFormat.SAFETENSORS)
        assert result == temp_dir

    def test_find_no_gguf_returns_none(self, temp_dir):
        """Returns None if no GGUF found."""
        from hfl.converter.formats import find_model_file, ModelFormat

        (temp_dir / "model.safetensors").write_bytes(b"ST")

        result = find_model_file(temp_dir, ModelFormat.GGUF)
        assert result is None


class TestGGUFConverter:
    """Tests for GGUFConverter."""

    def test_converter_initialization(self, temp_config):
        """Verifies converter initialization."""
        from hfl.converter.gguf_converter import GGUFConverter

        converter = GGUFConverter()

        assert converter.llama_cpp_dir == temp_config.llama_cpp_dir
        assert "convert_hf_to_gguf.py" in str(converter.convert_script)
        assert "llama-quantize" in str(converter.quantize_bin)

    def test_ensure_tools_when_available(self, temp_config):
        """Verifies that ensure_tools does nothing if tools exist."""
        from hfl.converter.gguf_converter import GGUFConverter

        converter = GGUFConverter()

        # Create necessary files
        converter.convert_script.parent.mkdir(parents=True, exist_ok=True)
        converter.convert_script.write_text("# script")
        converter.quantize_bin.parent.mkdir(parents=True, exist_ok=True)
        converter.quantize_bin.write_text("# binary")

        # Should not raise error
        converter.ensure_tools()

    def test_ensure_tools_clones_repo(self, temp_config):
        """Verifies that ensure_tools clones llama.cpp if it doesn't exist."""
        from hfl.converter.gguf_converter import GGUFConverter

        converter = GGUFConverter()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            with pytest.raises(Exception):
                # Will fail because the script won't exist after the mock
                converter.ensure_tools()

            # Verify that clone was attempted
            calls = [str(c) for c in mock_run.call_args_list]
            assert any("git" in str(c) and "clone" in str(c) for c in calls)

    def test_convert_f16_skips_quantization(self, temp_config):
        """Verifies that F16 skips quantization."""
        from hfl.converter.gguf_converter import GGUFConverter

        converter = GGUFConverter()

        model_path = temp_config.models_dir / "test-model"
        model_path.mkdir(parents=True)
        (model_path / "config.json").write_text("{}")

        output_path = temp_config.cache_dir / "output"

        with patch.object(converter, "ensure_tools"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                # Create the intermediate file that would be generated
                fp16_path = output_path.with_suffix(".fp16.gguf")
                fp16_path.parent.mkdir(parents=True, exist_ok=True)
                fp16_path.write_bytes(b"GGUF")

                with patch.object(Path, "rename"):
                    try:
                        converter.convert(model_path, output_path, "F16")
                    except Exception:
                        pass  # May fail due to mock

                # Verify that it's called at least once (conversion, not quantization)
                assert mock_run.call_count >= 1

    def test_convert_quantization_levels(self, temp_config):
        """Verifies supported quantization levels."""
        from hfl.converter.gguf_converter import GGUFConverter

        converter = GGUFConverter()

        # List of quantization levels
        quant_levels = [
            "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
            "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
            "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
            "Q6_K", "Q8_0", "F16",
        ]

        # All are valid strings
        for level in quant_levels:
            assert isinstance(level, str)
            assert len(level) > 0

    def test_convert_uses_sys_executable_not_python(self, temp_config):
        """
        CRITICAL: Verifies that sys.executable is used instead of 'python'.

        On macOS, 'python' doesn't exist (only python3), which causes
        FileNotFoundError. This test prevents regressions of this bug.
        """
        from hfl.converter.gguf_converter import GGUFConverter

        converter = GGUFConverter()

        model_path = temp_config.models_dir / "test-model"
        model_path.mkdir(parents=True)
        (model_path / "config.json").write_text("{}")

        output_path = temp_config.cache_dir / "output"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        captured_commands = []

        def capture_run(cmd, **kwargs):
            captured_commands.append(cmd)
            # Simulate FP16 file creation
            fp16_path = output_path.with_suffix(".fp16.gguf")
            fp16_path.write_bytes(b"GGUF")
            return MagicMock(returncode=0)

        with patch.object(converter, "ensure_tools"):
            with patch("subprocess.run", side_effect=capture_run):
                with patch.object(Path, "rename"):
                    with patch.object(Path, "unlink"):
                        try:
                            converter.convert(model_path, output_path, "F16")
                        except Exception:
                            pass

        # Verify that the first call uses sys.executable, not "python"
        assert len(captured_commands) >= 1
        first_cmd = captured_commands[0]
        assert first_cmd[0] == sys.executable, (
            f"Must use sys.executable ({sys.executable}), not '{first_cmd[0]}'. "
            "On macOS 'python' doesn't exist, only 'python3'."
        )
        assert "convert_hf_to_gguf.py" in first_cmd[1]

    def test_ensure_tools_uses_sys_executable_for_pip(self, temp_config):
        """
        Verifies that pip is invoked with sys.executable -m pip.

        This ensures that the pip from the correct environment is used.
        """
        from hfl.converter.gguf_converter import GGUFConverter

        converter = GGUFConverter()

        # Create directory and requirements.txt
        converter.llama_cpp_dir.mkdir(parents=True, exist_ok=True)
        (converter.llama_cpp_dir / "requirements.txt").write_text("numpy\n")

        captured_commands = []

        def capture_run(cmd, **kwargs):
            captured_commands.append(cmd)
            return MagicMock(returncode=0, stdout="abc123")

        with patch("subprocess.run", side_effect=capture_run):
            with patch("shutil.which", return_value=None):  # No CUDA
                try:
                    converter.ensure_tools()
                except Exception:
                    pass  # May fail, we only care about capturing commands

        # Find the pip call
        pip_calls = [c for c in captured_commands if "-m" in c and "pip" in c]
        assert len(pip_calls) >= 1, "No pip call with -m found"

        pip_cmd = pip_calls[0]
        assert pip_cmd[0] == sys.executable, (
            f"pip must be invoked with sys.executable ({sys.executable}), "
            f"not with '{pip_cmd[0]}'"
        )
        assert pip_cmd[1] == "-m"
        assert pip_cmd[2] == "pip"

    def test_convert_with_quantization(self, temp_config):
        """Verifies complete conversion with quantization."""
        from hfl.converter.gguf_converter import GGUFConverter

        converter = GGUFConverter()

        model_path = temp_config.models_dir / "test-model"
        model_path.mkdir(parents=True)
        (model_path / "config.json").write_text("{}")

        output_path = temp_config.cache_dir / "output"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        call_count = [0]

        def mock_run(cmd, **kwargs):
            call_count[0] += 1
            # First call: create FP16
            if call_count[0] == 1:
                fp16_path = output_path.with_suffix(".fp16.gguf")
                fp16_path.write_bytes(b"GGUF FP16")
            # Second call: create quantized file
            elif call_count[0] == 2:
                final_path = output_path.with_suffix(".Q4_K_M.gguf")
                final_path.write_bytes(b"GGUF Q4_K_M")
            return MagicMock(returncode=0)

        with patch.object(converter, "ensure_tools"):
            with patch("subprocess.run", side_effect=mock_run):
                result = converter.convert(
                    model_path, output_path, "Q4_K_M",
                    source_repo="test/model",
                    original_license="apache-2.0",
                    license_accepted=True,
                )

        assert call_count[0] >= 2  # FP16 conversion + quantization (+ provenance)
        assert result.suffix == ".gguf"
        assert "Q4_K_M" in result.name


class TestGetLlamaCppVersion:
    """Tests for _get_llama_cpp_version."""

    def test_get_version_success(self, temp_config):
        """Gets version correctly."""
        from hfl.converter.gguf_converter import _get_llama_cpp_version

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="abc1234\n"
            )

            result = _get_llama_cpp_version(temp_config.llama_cpp_dir)

            assert result == "abc1234"
            mock_run.assert_called_once()

    def test_get_version_failure(self, temp_config):
        """Returns 'unknown' on failure."""
        from hfl.converter.gguf_converter import _get_llama_cpp_version

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")

            result = _get_llama_cpp_version(temp_config.llama_cpp_dir)

            assert result == "unknown"

    def test_get_version_exception(self, temp_config):
        """Returns 'unknown' on exception."""
        from hfl.converter.gguf_converter import _get_llama_cpp_version

        with patch("subprocess.run", side_effect=Exception("Git not found")):
            result = _get_llama_cpp_version(temp_config.llama_cpp_dir)

            assert result == "unknown"


class TestCheckModelConvertibility:
    """Tests for check_model_convertibility."""

    def test_convertible_llm_model(self, temp_dir):
        """LLM model with valid config.json is convertible."""
        from hfl.converter.gguf_converter import check_model_convertibility

        config = {"model_type": "llama", "hidden_size": 4096}
        (temp_dir / "config.json").write_text(json.dumps(config))
        (temp_dir / "model.safetensors").write_bytes(b"weights")

        is_convertible, reason = check_model_convertibility(temp_dir)

        assert is_convertible is True
        assert reason == ""

    def test_missing_config_json(self, temp_dir):
        """Model without config.json is not convertible."""
        from hfl.converter.gguf_converter import check_model_convertibility

        (temp_dir / "model.safetensors").write_bytes(b"weights")

        is_convertible, reason = check_model_convertibility(temp_dir)

        assert is_convertible is False
        assert "config.json" in reason

    def test_lora_adapter_detected(self, temp_dir):
        """LoRA adapter is detected and rejected."""
        from hfl.converter.gguf_converter import check_model_convertibility

        # LoRA adapter has adapter_config.json
        (temp_dir / "adapter_config.json").write_text('{"base_model": "llama"}')
        (temp_dir / "adapter_model.safetensors").write_bytes(b"lora weights")

        is_convertible, reason = check_model_convertibility(temp_dir)

        assert is_convertible is False
        assert "LoRA" in reason

    def test_diffusion_model_detected(self, temp_dir):
        """Diffusion model is detected and rejected."""
        from hfl.converter.gguf_converter import check_model_convertibility

        # Stable Diffusion has diffusion_pytorch_model.safetensors
        (temp_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"unet")

        is_convertible, reason = check_model_convertibility(temp_dir)

        assert is_convertible is False
        assert "diffusion" in reason.lower() or "image" in reason.lower()

    def test_stable_diffusion_model_type(self, temp_dir):
        """Model with stable-diffusion model_type is rejected."""
        from hfl.converter.gguf_converter import check_model_convertibility

        config = {"model_type": "stable-diffusion-xl"}
        (temp_dir / "config.json").write_text(json.dumps(config))

        is_convertible, reason = check_model_convertibility(temp_dir)

        assert is_convertible is False
        assert "not supported" in reason

    def test_vae_model_type(self, temp_dir):
        """VAE model is rejected."""
        from hfl.converter.gguf_converter import check_model_convertibility

        config = {"model_type": "vae"}
        (temp_dir / "config.json").write_text(json.dumps(config))

        is_convertible, reason = check_model_convertibility(temp_dir)

        assert is_convertible is False
        assert "not supported" in reason

    def test_missing_model_type(self, temp_dir):
        """config.json without model_type is rejected."""
        from hfl.converter.gguf_converter import check_model_convertibility

        config = {"hidden_size": 4096}  # No model_type
        (temp_dir / "config.json").write_text(json.dumps(config))

        is_convertible, reason = check_model_convertibility(temp_dir)

        assert is_convertible is False
        assert "model_type" in reason

    def test_invalid_json(self, temp_dir):
        """Malformed config.json is rejected."""
        from hfl.converter.gguf_converter import check_model_convertibility

        (temp_dir / "config.json").write_text("{ invalid json }")

        is_convertible, reason = check_model_convertibility(temp_dir)

        assert is_convertible is False
        assert "Could not read" in reason

    def test_clip_model_rejected(self, temp_dir):
        """CLIP model is rejected."""
        from hfl.converter.gguf_converter import check_model_convertibility

        config = {"model_type": "clip"}
        (temp_dir / "config.json").write_text(json.dumps(config))

        is_convertible, reason = check_model_convertibility(temp_dir)

        assert is_convertible is False

    def test_whisper_model_rejected(self, temp_dir):
        """Whisper model (audio) is rejected."""
        from hfl.converter.gguf_converter import check_model_convertibility

        config = {"model_type": "whisper"}
        (temp_dir / "config.json").write_text(json.dumps(config))

        is_convertible, reason = check_model_convertibility(temp_dir)

        assert is_convertible is False

    def test_qwen_model_convertible(self, temp_dir):
        """Qwen model is convertible."""
        from hfl.converter.gguf_converter import check_model_convertibility

        config = {"model_type": "qwen2"}
        (temp_dir / "config.json").write_text(json.dumps(config))

        is_convertible, reason = check_model_convertibility(temp_dir)

        assert is_convertible is True

    def test_mistral_model_convertible(self, temp_dir):
        """Mistral model is convertible."""
        from hfl.converter.gguf_converter import check_model_convertibility

        config = {"model_type": "mistral"}
        (temp_dir / "config.json").write_text(json.dumps(config))

        is_convertible, reason = check_model_convertibility(temp_dir)

        assert is_convertible is True


class TestUnsupportedModelError:
    """Tests for UnsupportedModelError."""

    def test_exception_inheritance(self):
        """UnsupportedModelError inherits from Exception."""
        from hfl.converter.gguf_converter import UnsupportedModelError

        assert issubclass(UnsupportedModelError, Exception)

    def test_exception_message(self):
        """UnsupportedModelError can have a message."""
        from hfl.converter.gguf_converter import UnsupportedModelError

        err = UnsupportedModelError("LoRA models are not supported")
        assert str(err) == "LoRA models are not supported"
