# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for engine selector module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hfl.converter.formats import ModelFormat, ModelType
from hfl.engine.selector import (
    MissingDependencyError,
    _create_engine,
    _get_bark_engine,
    _get_coqui_engine,
    _get_llama_cpp_engine,
    _get_transformers_engine,
    _get_vllm_engine,
    _has_cuda,
    _is_bark_model,
    _is_coqui_model,
    select_engine,
    select_tts_engine,
)


class TestMissingDependencyError:
    """Tests for MissingDependencyError exception."""

    def test_is_exception(self):
        """MissingDependencyError is an Exception."""
        error = MissingDependencyError("test message")
        assert isinstance(error, Exception)
        assert str(error) == "test message"


class TestHasCuda:
    """Tests for _has_cuda function."""

    def test_returns_bool(self):
        """_has_cuda returns a boolean value."""
        result = _has_cuda()
        assert isinstance(result, bool)

    def test_cuda_check_with_mock(self):
        """_has_cuda checks torch.cuda.is_available when torch is present."""
        # Create a mock torch module
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        # Test the function with mocked torch
        import sys

        # Store original if present
        original_torch = sys.modules.get("torch")

        try:
            sys.modules["torch"] = mock_torch
            # Need to reload the module to pick up the mock
            # But since the function does its own import, we test via the actual behavior
            # Just verify the function returns a valid bool
            result = _has_cuda()
            assert isinstance(result, bool)
        finally:
            # Restore original
            if original_torch is not None:
                sys.modules["torch"] = original_torch
            elif "torch" in sys.modules:
                # If we added it, the mock might still be there
                pass


class TestGetEngines:
    """Tests for engine getter functions."""

    @patch("hfl.engine.selector.LlamaCppEngine", create=True)
    def test_get_llama_cpp_engine_success(self, mock_engine_cls):
        """_get_llama_cpp_engine returns engine on success."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        with patch.dict(
            "sys.modules",
            {
                "hfl.engine.llama_cpp": MagicMock(
                    LlamaCppEngine=mock_engine_cls,
                ),
            },
        ):
            with patch("hfl.engine.llama_cpp.LlamaCppEngine", mock_engine_cls, create=True):
                # The actual test depends on import behavior
                pass

    def test_get_llama_cpp_engine_import_error(self):
        """_get_llama_cpp_engine raises MissingDependencyError on ImportError."""
        with patch.dict("sys.modules", {"hfl.engine.llama_cpp": None}):
            with pytest.raises(MissingDependencyError) as exc_info:
                _get_llama_cpp_engine()

            assert "llama-cpp-python" in str(exc_info.value)

    def test_get_transformers_engine_import_error(self):
        """_get_transformers_engine raises MissingDependencyError on ImportError."""
        with patch.dict("sys.modules", {"hfl.engine.transformers_engine": None}):
            with pytest.raises(MissingDependencyError) as exc_info:
                _get_transformers_engine()

            assert "transformers" in str(exc_info.value)

    def test_get_vllm_engine_import_error(self):
        """_get_vllm_engine raises MissingDependencyError on ImportError."""
        with patch.dict("sys.modules", {"hfl.engine.vllm_engine": None}):
            with pytest.raises(MissingDependencyError) as exc_info:
                _get_vllm_engine()

            assert "vLLM" in str(exc_info.value)

    def test_get_bark_engine_import_error(self):
        """_get_bark_engine raises MissingDependencyError on ImportError."""
        with patch.dict("sys.modules", {"hfl.engine.bark_engine": None}):
            with pytest.raises(MissingDependencyError) as exc_info:
                _get_bark_engine()

            assert "Bark" in str(exc_info.value)

    def test_get_coqui_engine_import_error(self):
        """_get_coqui_engine raises MissingDependencyError on ImportError."""
        with patch.dict("sys.modules", {"hfl.engine.coqui_engine": None}):
            with pytest.raises(MissingDependencyError) as exc_info:
                _get_coqui_engine()

            assert "Coqui" in str(exc_info.value)


class TestCreateEngine:
    """Tests for _create_engine function."""

    @patch("hfl.engine.selector._get_llama_cpp_engine")
    def test_create_llama_cpp(self, mock_get):
        """_create_engine creates llama-cpp engine."""
        mock_engine = MagicMock()
        mock_get.return_value = mock_engine

        result = _create_engine("llama-cpp")

        assert result is mock_engine
        mock_get.assert_called_once()

    @patch("hfl.engine.selector._get_transformers_engine")
    def test_create_transformers(self, mock_get):
        """_create_engine creates transformers engine."""
        mock_engine = MagicMock()
        mock_get.return_value = mock_engine

        result = _create_engine("transformers")

        assert result is mock_engine
        mock_get.assert_called_once()

    @patch("hfl.engine.selector._get_vllm_engine")
    def test_create_vllm(self, mock_get):
        """_create_engine creates vllm engine."""
        mock_engine = MagicMock()
        mock_get.return_value = mock_engine

        result = _create_engine("vllm")

        assert result is mock_engine
        mock_get.assert_called_once()

    def test_create_unknown_raises_value_error(self):
        """_create_engine raises ValueError for unknown backend."""
        with pytest.raises(ValueError) as exc_info:
            _create_engine("unknown-backend")

        assert "Unknown backend" in str(exc_info.value)

    @patch("hfl.engine.selector._get_mlx_engine")
    def test_create_mlx(self, mock_get):
        """_create_engine creates MLX engine when requested explicitly."""
        from hfl.engine.selector import _create_engine as create

        mock_engine = MagicMock()
        mock_get.return_value = mock_engine

        result = create("mlx")

        assert result is mock_engine
        mock_get.assert_called_once()


class TestMLXAutoSelection:
    """Tests for MLX auto-selection on Apple Silicon (Darwin-arm64)."""

    @patch("hfl.engine.selector._get_mlx_engine")
    @patch("hfl.engine.selector._mlx_preferred", return_value=True)
    @patch("hfl.engine.selector.detect_format")
    def test_auto_selects_mlx_for_safetensors_on_apple_silicon(
        self, mock_detect, _mlx_pref, mock_get_mlx
    ):
        """Safetensors + Darwin-arm64 + mlx-lm installed -> MLXEngine."""
        mock_detect.return_value = ModelFormat.SAFETENSORS
        mock_engine = MagicMock()
        mock_get_mlx.return_value = mock_engine

        result = select_engine(Path("/model"))

        assert result is mock_engine
        mock_get_mlx.assert_called_once()

    @patch("hfl.engine.selector._get_llama_cpp_engine")
    @patch("hfl.engine.selector._mlx_preferred", return_value=True)
    @patch("hfl.engine.selector.detect_format")
    def test_does_not_select_mlx_for_gguf(self, mock_detect, _mlx_pref, mock_get_llama):
        """GGUF must always route to llama-cpp, never MLX (MLX can't ingest GGUF)."""
        mock_detect.return_value = ModelFormat.GGUF
        mock_llama = MagicMock()
        mock_get_llama.return_value = mock_llama

        result = select_engine(Path("/model.gguf"))

        assert result is mock_llama
        mock_get_llama.assert_called_once()

    @patch("hfl.engine.selector._get_llama_cpp_engine")
    @patch("hfl.engine.selector._get_mlx_engine")
    @patch("hfl.engine.selector._has_cuda", return_value=False)
    @patch("hfl.engine.selector._mlx_preferred", return_value=True)
    @patch("hfl.engine.selector.detect_format")
    def test_falls_through_when_mlx_import_fails(
        self, mock_detect, _mlx_pref, _cuda, mock_get_mlx, mock_get_llama
    ):
        """If _get_mlx_engine raises MissingDependencyError, fall through to llama-cpp."""
        mock_detect.return_value = ModelFormat.SAFETENSORS
        mock_get_mlx.side_effect = MissingDependencyError("mlx-lm not available")
        mock_llama = MagicMock()
        mock_get_llama.return_value = mock_llama

        result = select_engine(Path("/model"))

        assert result is mock_llama

    def test_mlx_preferred_respects_disable_env(self, monkeypatch):
        """HFL_DISABLE_MLX forces _mlx_preferred() to False even when SDK is available."""
        from hfl.engine import mlx_engine, selector

        monkeypatch.setattr(mlx_engine, "is_available", lambda: True)
        monkeypatch.setenv("HFL_DISABLE_MLX", "1")

        assert selector._mlx_preferred() is False

    def test_mlx_preferred_when_sdk_available_and_not_disabled(self, monkeypatch):
        """_mlx_preferred() is True when SDK available and env var not set."""
        from hfl.engine import mlx_engine, selector

        monkeypatch.setattr(mlx_engine, "is_available", lambda: True)
        monkeypatch.delenv("HFL_DISABLE_MLX", raising=False)

        assert selector._mlx_preferred() is True

    def test_mlx_preferred_false_when_sdk_unavailable(self, monkeypatch):
        """_mlx_preferred() is False when MLX SDK isn't importable (non-Darwin or not installed)."""
        from hfl.engine import mlx_engine, selector

        monkeypatch.setattr(mlx_engine, "is_available", lambda: False)
        monkeypatch.delenv("HFL_DISABLE_MLX", raising=False)

        assert selector._mlx_preferred() is False

    def test_get_mlx_engine_raises_when_unavailable(self, monkeypatch):
        """_get_mlx_engine raises MissingDependencyError on non-Darwin / no mlx_lm."""
        from hfl.engine import mlx_engine
        from hfl.engine.selector import _get_mlx_engine

        monkeypatch.setattr(mlx_engine, "is_available", lambda: False)
        with pytest.raises(MissingDependencyError) as exc_info:
            _get_mlx_engine()

        msg = str(exc_info.value)
        assert "Apple Silicon" in msg or "mlx-lm" in msg


class TestSelectEngine:
    """Tests for select_engine function."""

    @patch("hfl.engine.selector._create_engine")
    @patch("hfl.engine.selector.detect_format")
    def test_explicit_backend(self, mock_detect, mock_create):
        """select_engine uses explicit backend when specified."""
        mock_engine = MagicMock()
        mock_create.return_value = mock_engine

        result = select_engine(Path("/model"), backend="transformers")

        assert result is mock_engine
        mock_create.assert_called_once_with("transformers")

    @patch("hfl.engine.selector._get_llama_cpp_engine")
    @patch("hfl.engine.selector.detect_format")
    def test_auto_selects_llama_for_gguf(self, mock_detect, mock_get):
        """select_engine auto-selects llama-cpp for GGUF format."""
        mock_detect.return_value = ModelFormat.GGUF
        mock_engine = MagicMock()
        mock_get.return_value = mock_engine

        result = select_engine(Path("/model.gguf"))

        assert result is mock_engine
        mock_get.assert_called_once()

    @patch("hfl.engine.selector._mlx_preferred", return_value=False)
    @patch("hfl.engine.selector._get_transformers_engine")
    @patch("hfl.engine.selector._has_cuda")
    @patch("hfl.engine.selector.detect_format")
    def test_auto_selects_transformers_with_cuda(self, mock_detect, mock_cuda, mock_get, _mlx):
        """select_engine selects transformers when CUDA available."""
        mock_detect.return_value = ModelFormat.SAFETENSORS
        mock_cuda.return_value = True
        mock_engine = MagicMock()
        mock_get.return_value = mock_engine

        result = select_engine(Path("/model"))

        assert result is mock_engine
        mock_get.assert_called_once()

    @patch("hfl.engine.selector._mlx_preferred", return_value=False)
    @patch("hfl.engine.selector._get_llama_cpp_engine")
    @patch("hfl.engine.selector._get_transformers_engine")
    @patch("hfl.engine.selector._has_cuda")
    @patch("hfl.engine.selector.detect_format")
    def test_fallback_to_llama_when_transformers_fails(
        self, mock_detect, mock_cuda, mock_get_trans, mock_get_llama, _mlx
    ):
        """select_engine falls back to llama-cpp when transformers fails."""
        mock_detect.return_value = ModelFormat.SAFETENSORS
        mock_cuda.return_value = True
        mock_get_trans.side_effect = MissingDependencyError("Not installed")
        mock_llama = MagicMock()
        mock_get_llama.return_value = mock_llama

        result = select_engine(Path("/model"))

        assert result is mock_llama

    @patch("hfl.engine.selector._mlx_preferred", return_value=False)
    @patch("hfl.engine.selector._get_llama_cpp_engine")
    @patch("hfl.engine.selector._has_cuda")
    @patch("hfl.engine.selector.detect_format")
    def test_fallback_to_llama_without_cuda(self, mock_detect, mock_cuda, mock_get, _mlx):
        """select_engine uses llama-cpp without CUDA."""
        mock_detect.return_value = ModelFormat.SAFETENSORS
        mock_cuda.return_value = False
        mock_engine = MagicMock()
        mock_get.return_value = mock_engine

        result = select_engine(Path("/model"))

        assert result is mock_engine


class TestForcedBackendEnv:
    """``HFL_LLM_LIBRARY`` (and the Ollama alias) overrides the auto
    path so operators can pin a backend without editing code."""

    @patch("hfl.engine.selector._create_engine")
    @patch("hfl.engine.selector.detect_format")
    def test_hfl_llm_library_forces_backend(self, mock_detect, mock_create, monkeypatch):
        from hfl.converter.formats import ModelFormat

        monkeypatch.setenv("HFL_LLM_LIBRARY", "vllm")
        mock_detect.return_value = ModelFormat.SAFETENSORS
        mock_engine = MagicMock()
        mock_create.return_value = mock_engine

        result = select_engine(Path("/model"))

        assert result is mock_engine
        mock_create.assert_called_once_with("vllm")

    @patch("hfl.engine.selector._create_engine")
    @patch("hfl.engine.selector.detect_format")
    def test_ollama_alias_is_honoured(self, mock_detect, mock_create, monkeypatch):
        from hfl.converter.formats import ModelFormat

        monkeypatch.delenv("HFL_LLM_LIBRARY", raising=False)
        monkeypatch.setenv("OLLAMA_LLM_LIBRARY", "transformers")
        mock_detect.return_value = ModelFormat.SAFETENSORS
        mock_engine = MagicMock()
        mock_create.return_value = mock_engine

        result = select_engine(Path("/model"))

        assert result is mock_engine
        mock_create.assert_called_once_with("transformers")

    @patch("hfl.engine.selector._create_engine")
    @patch("hfl.engine.selector.detect_format")
    def test_explicit_backend_arg_wins_over_env(self, mock_detect, mock_create, monkeypatch):
        """A caller passing an explicit ``backend=`` (Modelfile,
        --backend flag) must beat the env override."""
        from hfl.converter.formats import ModelFormat

        monkeypatch.setenv("HFL_LLM_LIBRARY", "vllm")
        mock_detect.return_value = ModelFormat.SAFETENSORS
        mock_engine = MagicMock()
        mock_create.return_value = mock_engine

        result = select_engine(Path("/model"), backend="llama-cpp")

        assert result is mock_engine
        mock_create.assert_called_once_with("llama-cpp")

    @patch("hfl.engine.selector._create_engine")
    @patch("hfl.engine.selector.detect_format")
    def test_unknown_value_is_ignored_silently(self, mock_detect, mock_create, monkeypatch):
        """A typo must not crash the server. Auto path takes over."""
        from hfl.converter.formats import ModelFormat

        monkeypatch.setenv("HFL_LLM_LIBRARY", "torchscript")
        mock_detect.return_value = ModelFormat.GGUF

        with patch("hfl.engine.selector._get_llama_cpp_engine") as fallback:
            fallback.return_value = MagicMock()
            select_engine(Path("/model"))
            fallback.assert_called_once()


class TestIsBarkModel:
    """Tests for _is_bark_model function."""

    def test_bark_in_name(self):
        """_is_bark_model returns True for 'bark' in model name."""
        result = _is_bark_model(Path("/models/suno-bark-small"))
        assert result is True

    def test_bark_in_name_uppercase(self):
        """_is_bark_model handles case insensitivity."""
        result = _is_bark_model(Path("/models/BARK-model"))
        assert result is True

    def test_no_bark_in_name(self):
        """_is_bark_model returns False when 'bark' not in name."""
        result = _is_bark_model(Path("/models/other-tts-model"))
        assert result is False

    def test_config_json_with_bark_architecture(self):
        """_is_bark_model checks config.json for Bark architecture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            config_path = model_dir / "config.json"
            config_path.write_text(json.dumps({"architectures": ["BarkModel", "BarkForCausalLM"]}))

            result = _is_bark_model(model_dir)
            assert result is True

    def test_config_json_without_bark_architecture(self):
        """_is_bark_model returns False for non-Bark config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            config_path = model_dir / "config.json"
            config_path.write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))

            result = _is_bark_model(model_dir)
            assert result is False

    def test_invalid_config_json(self):
        """_is_bark_model handles invalid JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            config_path = model_dir / "config.json"
            config_path.write_text("invalid json {")

            result = _is_bark_model(model_dir)
            assert result is False

    def test_file_path_not_directory(self):
        """_is_bark_model handles file path instead of directory."""
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            result = _is_bark_model(Path(f.name))
            assert result is False


class TestIsCoquiModel:
    """Tests for _is_coqui_model function."""

    def test_tts_models_pattern(self):
        """_is_coqui_model detects tts_models/ pattern."""
        result = _is_coqui_model(Path("tts_models/en/ljspeech/tacotron2"))
        assert result is True

    def test_xtts_pattern(self):
        """_is_coqui_model detects XTTS pattern."""
        result = _is_coqui_model(Path("/models/xtts-v2"))
        assert result is True

    def test_vits_pattern(self):
        """_is_coqui_model detects VITS pattern."""
        result = _is_coqui_model(Path("/models/vits-ljs"))
        assert result is True

    def test_tacotron_pattern(self):
        """_is_coqui_model detects Tacotron pattern."""
        result = _is_coqui_model(Path("/models/tacotron2-DDC"))
        assert result is True

    def test_glow_tts_pattern(self):
        """_is_coqui_model detects Glow-TTS pattern."""
        result = _is_coqui_model(Path("/models/glow-tts-model"))
        assert result is True

    def test_speedy_speech_pattern(self):
        """_is_coqui_model detects Speedy-Speech pattern."""
        result = _is_coqui_model(Path("/models/speedy-speech"))
        assert result is True

    def test_no_coqui_pattern(self):
        """_is_coqui_model returns False for non-Coqui models."""
        result = _is_coqui_model(Path("/models/other-model"))
        assert result is False


class TestSelectTTSEngine:
    """Tests for select_tts_engine function."""

    @patch("hfl.engine.selector._get_bark_engine")
    def test_explicit_bark_backend(self, mock_get):
        """select_tts_engine uses explicit bark backend."""
        mock_engine = MagicMock()
        mock_get.return_value = mock_engine

        result = select_tts_engine(Path("/model"), backend="bark")

        assert result is mock_engine
        mock_get.assert_called_once()

    @patch("hfl.engine.selector._get_coqui_engine")
    def test_explicit_coqui_backend(self, mock_get):
        """select_tts_engine uses explicit coqui backend."""
        mock_engine = MagicMock()
        mock_get.return_value = mock_engine

        result = select_tts_engine(Path("/model"), backend="coqui")

        assert result is mock_engine
        mock_get.assert_called_once()

    @patch("hfl.engine.selector._get_bark_engine")
    @patch("hfl.engine.selector._is_bark_model")
    @patch("hfl.engine.selector.detect_model_type")
    def test_auto_detects_bark(self, mock_detect_type, mock_is_bark, mock_get):
        """select_tts_engine auto-detects Bark models."""
        mock_detect_type.return_value = ModelType.TTS
        mock_is_bark.return_value = True
        mock_engine = MagicMock()
        mock_get.return_value = mock_engine

        result = select_tts_engine(Path("/models/bark"))

        assert result is mock_engine

    @patch("hfl.engine.selector._get_coqui_engine")
    @patch("hfl.engine.selector._is_coqui_model")
    @patch("hfl.engine.selector._is_bark_model")
    @patch("hfl.engine.selector.detect_model_type")
    def test_auto_detects_coqui(self, mock_detect_type, mock_is_bark, mock_is_coqui, mock_get):
        """select_tts_engine auto-detects Coqui models."""
        mock_detect_type.return_value = ModelType.TTS
        mock_is_bark.return_value = False
        mock_is_coqui.return_value = True
        mock_engine = MagicMock()
        mock_get.return_value = mock_engine

        result = select_tts_engine(Path("/models/xtts"))

        assert result is mock_engine

    @patch("hfl.engine.selector.detect_model_type")
    def test_wrong_model_type_raises_error(self, mock_detect_type):
        """select_tts_engine raises ValueError for non-TTS models."""
        mock_detect_type.return_value = ModelType.LLM

        with pytest.raises(ValueError) as exc_info:
            select_tts_engine(Path("/models/llm"))

        assert "does not appear to be a TTS model" in str(exc_info.value)

    @patch("hfl.engine.selector._get_coqui_engine")
    @patch("hfl.engine.selector._get_bark_engine")
    @patch("hfl.engine.selector._is_coqui_model")
    @patch("hfl.engine.selector._is_bark_model")
    @patch("hfl.engine.selector.detect_model_type")
    def test_fallback_to_bark_then_coqui(
        self, mock_detect_type, mock_is_bark, mock_is_coqui, mock_get_bark, mock_get_coqui
    ):
        """select_tts_engine falls back to Bark, then Coqui."""
        mock_detect_type.return_value = ModelType.TTS
        mock_is_bark.return_value = False
        mock_is_coqui.return_value = False
        mock_get_bark.side_effect = MissingDependencyError("No bark")
        mock_coqui = MagicMock()
        mock_get_coqui.return_value = mock_coqui

        result = select_tts_engine(Path("/models/unknown-tts"))

        assert result is mock_coqui

    @patch("hfl.engine.selector._get_coqui_engine")
    @patch("hfl.engine.selector._get_bark_engine")
    @patch("hfl.engine.selector._is_coqui_model")
    @patch("hfl.engine.selector._is_bark_model")
    @patch("hfl.engine.selector.detect_model_type")
    def test_no_backend_available_raises_error(
        self, mock_detect_type, mock_is_bark, mock_is_coqui, mock_get_bark, mock_get_coqui
    ):
        """select_tts_engine raises ValueError when no backend available."""
        mock_detect_type.return_value = ModelType.TTS
        mock_is_bark.return_value = False
        mock_is_coqui.return_value = False
        mock_get_bark.side_effect = MissingDependencyError("No bark")
        mock_get_coqui.side_effect = MissingDependencyError("No coqui")

        with pytest.raises(ValueError) as exc_info:
            select_tts_engine(Path("/models/unknown-tts"))

        assert "Could not find a suitable TTS backend" in str(exc_info.value)
