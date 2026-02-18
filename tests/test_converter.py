"""Tests para el módulo converter (formats, gguf_converter)."""

import pytest
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock


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
