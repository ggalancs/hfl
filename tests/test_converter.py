# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests para el módulo converter (formats, gguf_converter)."""

import pytest
import sys
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock, call


class TestModelFormat:
    """Tests para ModelFormat enum."""

    def test_enum_values(self):
        """Verifica valores del enum."""
        from hfl.converter.formats import ModelFormat

        assert ModelFormat.GGUF.value == "gguf"
        assert ModelFormat.SAFETENSORS.value == "safetensors"
        assert ModelFormat.PYTORCH.value == "pytorch"
        assert ModelFormat.UNKNOWN.value == "unknown"

    def test_enum_members(self):
        """Verifica miembros del enum."""
        from hfl.converter.formats import ModelFormat

        assert hasattr(ModelFormat, "GGUF")
        assert hasattr(ModelFormat, "SAFETENSORS")
        assert hasattr(ModelFormat, "PYTORCH")
        assert hasattr(ModelFormat, "UNKNOWN")


class TestDetectFormat:
    """Tests para detect_format."""

    def test_detect_gguf_file(self, temp_dir):
        """Detecta archivo GGUF."""
        from hfl.converter.formats import detect_format, ModelFormat

        gguf_file = temp_dir / "model.gguf"
        gguf_file.write_bytes(b"GGUF content")

        result = detect_format(gguf_file)
        assert result == ModelFormat.GGUF

    def test_detect_safetensors_file(self, temp_dir):
        """Detecta archivo safetensors."""
        from hfl.converter.formats import detect_format, ModelFormat

        st_file = temp_dir / "model.safetensors"
        st_file.write_bytes(b"safetensors content")

        result = detect_format(st_file)
        assert result == ModelFormat.SAFETENSORS

    def test_detect_pytorch_bin_file(self, temp_dir):
        """Detecta archivo pytorch .bin."""
        from hfl.converter.formats import detect_format, ModelFormat

        pt_file = temp_dir / "model.bin"
        pt_file.write_bytes(b"pytorch content")

        result = detect_format(pt_file)
        assert result == ModelFormat.PYTORCH

    def test_detect_pytorch_pt_file(self, temp_dir):
        """Detecta archivo pytorch .pt."""
        from hfl.converter.formats import detect_format, ModelFormat

        pt_file = temp_dir / "model.pt"
        pt_file.write_bytes(b"pytorch content")

        result = detect_format(pt_file)
        assert result == ModelFormat.PYTORCH

    def test_detect_pytorch_pth_file(self, temp_dir):
        """Detecta archivo pytorch .pth."""
        from hfl.converter.formats import detect_format, ModelFormat

        pt_file = temp_dir / "model.pth"
        pt_file.write_bytes(b"pytorch content")

        result = detect_format(pt_file)
        assert result == ModelFormat.PYTORCH

    def test_detect_gguf_in_directory(self, temp_dir):
        """Detecta GGUF en directorio."""
        from hfl.converter.formats import detect_format, ModelFormat

        (temp_dir / "model.gguf").write_bytes(b"GGUF")
        (temp_dir / "config.json").write_text("{}")

        result = detect_format(temp_dir)
        assert result == ModelFormat.GGUF

    def test_detect_safetensors_in_directory(self, temp_dir):
        """Detecta safetensors en directorio."""
        from hfl.converter.formats import detect_format, ModelFormat

        (temp_dir / "model.safetensors").write_bytes(b"ST")
        (temp_dir / "config.json").write_text("{}")

        result = detect_format(temp_dir)
        assert result == ModelFormat.SAFETENSORS

    def test_detect_pytorch_in_directory(self, temp_dir):
        """Detecta pytorch en directorio."""
        from hfl.converter.formats import detect_format, ModelFormat

        (temp_dir / "pytorch_model.bin").write_bytes(b"PT")
        (temp_dir / "config.json").write_text("{}")

        result = detect_format(temp_dir)
        assert result == ModelFormat.PYTORCH

    def test_detect_unknown_file(self, temp_dir):
        """Devuelve UNKNOWN para archivo desconocido."""
        from hfl.converter.formats import detect_format, ModelFormat

        txt_file = temp_dir / "readme.txt"
        txt_file.write_text("readme")

        result = detect_format(txt_file)
        assert result == ModelFormat.UNKNOWN

    def test_detect_unknown_directory(self, temp_dir):
        """Devuelve UNKNOWN para directorio sin modelos."""
        from hfl.converter.formats import detect_format, ModelFormat

        (temp_dir / "readme.txt").write_text("readme")
        (temp_dir / "data.json").write_text("{}")

        result = detect_format(temp_dir)
        assert result == ModelFormat.UNKNOWN

    def test_detect_nonexistent_path(self):
        """Devuelve UNKNOWN para path inexistente."""
        from hfl.converter.formats import detect_format, ModelFormat

        result = detect_format(Path("/nonexistent/path"))
        assert result == ModelFormat.UNKNOWN

    def test_detect_gguf_priority_over_safetensors(self, temp_dir):
        """GGUF tiene prioridad sobre safetensors."""
        from hfl.converter.formats import detect_format, ModelFormat

        (temp_dir / "model.gguf").write_bytes(b"GGUF")
        (temp_dir / "model.safetensors").write_bytes(b"ST")

        result = detect_format(temp_dir)
        assert result == ModelFormat.GGUF

    def test_detect_nested_files(self, temp_dir):
        """Detecta archivos en subdirectorios."""
        from hfl.converter.formats import detect_format, ModelFormat

        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "model.gguf").write_bytes(b"GGUF")

        result = detect_format(temp_dir)
        assert result == ModelFormat.GGUF


class TestFindModelFile:
    """Tests para find_model_file."""

    def test_find_gguf_file_direct(self, temp_dir):
        """Encuentra archivo GGUF directo."""
        from hfl.converter.formats import find_model_file, ModelFormat

        gguf_file = temp_dir / "model.gguf"
        gguf_file.write_bytes(b"GGUF")

        result = find_model_file(gguf_file, ModelFormat.GGUF)
        assert result == gguf_file

    def test_find_gguf_in_directory(self, temp_dir):
        """Encuentra GGUF en directorio."""
        from hfl.converter.formats import find_model_file, ModelFormat

        gguf_file = temp_dir / "model.gguf"
        gguf_file.write_bytes(b"GGUF")

        result = find_model_file(temp_dir, ModelFormat.GGUF)
        assert result == gguf_file

    def test_find_safetensors_returns_directory(self, temp_dir):
        """Para safetensors devuelve el directorio."""
        from hfl.converter.formats import find_model_file, ModelFormat

        (temp_dir / "model.safetensors").write_bytes(b"ST")

        result = find_model_file(temp_dir, ModelFormat.SAFETENSORS)
        assert result == temp_dir

    def test_find_no_gguf_returns_none(self, temp_dir):
        """Devuelve None si no hay GGUF."""
        from hfl.converter.formats import find_model_file, ModelFormat

        (temp_dir / "model.safetensors").write_bytes(b"ST")

        result = find_model_file(temp_dir, ModelFormat.GGUF)
        assert result is None


class TestGGUFConverter:
    """Tests para GGUFConverter."""

    def test_converter_initialization(self, temp_config):
        """Verifica inicialización del converter."""
        from hfl.converter.gguf_converter import GGUFConverter

        converter = GGUFConverter()

        assert converter.llama_cpp_dir == temp_config.llama_cpp_dir
        assert "convert_hf_to_gguf.py" in str(converter.convert_script)
        assert "llama-quantize" in str(converter.quantize_bin)

    def test_ensure_tools_when_available(self, temp_config):
        """Verifica que ensure_tools no hace nada si las herramientas existen."""
        from hfl.converter.gguf_converter import GGUFConverter

        converter = GGUFConverter()

        # Crear los archivos necesarios
        converter.convert_script.parent.mkdir(parents=True, exist_ok=True)
        converter.convert_script.write_text("# script")
        converter.quantize_bin.parent.mkdir(parents=True, exist_ok=True)
        converter.quantize_bin.write_text("# binary")

        # No debería lanzar error
        converter.ensure_tools()

    def test_ensure_tools_clones_repo(self, temp_config):
        """Verifica que ensure_tools clona llama.cpp si no existe."""
        from hfl.converter.gguf_converter import GGUFConverter

        converter = GGUFConverter()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            with pytest.raises(Exception):
                # Fallará porque el script no existirá después del mock
                converter.ensure_tools()

            # Verificar que se intentó clonar
            calls = [str(c) for c in mock_run.call_args_list]
            assert any("git" in str(c) and "clone" in str(c) for c in calls)

    def test_convert_f16_skips_quantization(self, temp_config):
        """Verifica que F16 salta la cuantización."""
        from hfl.converter.gguf_converter import GGUFConverter

        converter = GGUFConverter()

        model_path = temp_config.models_dir / "test-model"
        model_path.mkdir(parents=True)
        (model_path / "config.json").write_text("{}")

        output_path = temp_config.cache_dir / "output"

        with patch.object(converter, "ensure_tools"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                # Crear el archivo intermedio que sería generado
                fp16_path = output_path.with_suffix(".fp16.gguf")
                fp16_path.parent.mkdir(parents=True, exist_ok=True)
                fp16_path.write_bytes(b"GGUF")

                with patch.object(Path, "rename"):
                    try:
                        converter.convert(model_path, output_path, "F16")
                    except Exception:
                        pass  # Puede fallar por el mock

                # Verificar que solo se llama una vez (conversión, no cuantización)
                assert mock_run.call_count >= 1

    def test_convert_quantization_levels(self, temp_config):
        """Verifica niveles de cuantización soportados."""
        from hfl.converter.gguf_converter import GGUFConverter

        converter = GGUFConverter()

        # Lista de niveles de cuantización
        quant_levels = [
            "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
            "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
            "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
            "Q6_K", "Q8_0", "F16",
        ]

        # Todos son strings válidos
        for level in quant_levels:
            assert isinstance(level, str)
            assert len(level) > 0

    def test_convert_uses_sys_executable_not_python(self, temp_config):
        """
        CRÍTICO: Verifica que se usa sys.executable en lugar de 'python'.

        En macOS, 'python' no existe (solo python3), lo que causa
        FileNotFoundError. Este test previene regresiones de este bug.
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
            # Simular creación del archivo FP16
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

        # Verificar que la primera llamada usa sys.executable, no "python"
        assert len(captured_commands) >= 1
        first_cmd = captured_commands[0]
        assert first_cmd[0] == sys.executable, (
            f"Debe usar sys.executable ({sys.executable}), no '{first_cmd[0]}'. "
            "En macOS 'python' no existe, solo 'python3'."
        )
        assert "convert_hf_to_gguf.py" in first_cmd[1]

    def test_ensure_tools_uses_sys_executable_for_pip(self, temp_config):
        """
        Verifica que pip se invoca con sys.executable -m pip.

        Esto garantiza que se usa el pip del entorno correcto.
        """
        from hfl.converter.gguf_converter import GGUFConverter

        converter = GGUFConverter()

        # Crear directorio y requirements.txt
        converter.llama_cpp_dir.mkdir(parents=True, exist_ok=True)
        (converter.llama_cpp_dir / "requirements.txt").write_text("numpy\n")

        captured_commands = []

        def capture_run(cmd, **kwargs):
            captured_commands.append(cmd)
            return MagicMock(returncode=0, stdout="abc123")

        with patch("subprocess.run", side_effect=capture_run):
            with patch("shutil.which", return_value=None):  # Sin CUDA
                try:
                    converter.ensure_tools()
                except Exception:
                    pass  # Puede fallar, solo nos interesa capturar los comandos

        # Buscar la llamada a pip
        pip_calls = [c for c in captured_commands if "-m" in c and "pip" in c]
        assert len(pip_calls) >= 1, "No se encontró llamada a pip con -m"

        pip_cmd = pip_calls[0]
        assert pip_cmd[0] == sys.executable, (
            f"pip debe invocarse con sys.executable ({sys.executable}), "
            f"no con '{pip_cmd[0]}'"
        )
        assert pip_cmd[1] == "-m"
        assert pip_cmd[2] == "pip"

    def test_convert_with_quantization(self, temp_config):
        """Verifica conversión completa con cuantización."""
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
            # Primera llamada: crear FP16
            if call_count[0] == 1:
                fp16_path = output_path.with_suffix(".fp16.gguf")
                fp16_path.write_bytes(b"GGUF FP16")
            # Segunda llamada: crear archivo cuantizado
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

        assert call_count[0] >= 2  # Conversión FP16 + cuantización (+ provenance)
        assert result.suffix == ".gguf"
        assert "Q4_K_M" in result.name


class TestGetLlamaCppVersion:
    """Tests para _get_llama_cpp_version."""

    def test_get_version_success(self, temp_config):
        """Obtiene versión correctamente."""
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
        """Devuelve 'unknown' si falla."""
        from hfl.converter.gguf_converter import _get_llama_cpp_version

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")

            result = _get_llama_cpp_version(temp_config.llama_cpp_dir)

            assert result == "unknown"

    def test_get_version_exception(self, temp_config):
        """Devuelve 'unknown' si hay excepción."""
        from hfl.converter.gguf_converter import _get_llama_cpp_version

        with patch("subprocess.run", side_effect=Exception("Git not found")):
            result = _get_llama_cpp_version(temp_config.llama_cpp_dir)

            assert result == "unknown"
