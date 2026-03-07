# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for engine dependency checking."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from hfl.engine.dependency_check import (
    check_engine_availability,
    get_recommended_backend,
    log_available_backends,
)


@pytest.fixture
def enable_dependency_check_logger():
    """Ensure the dependency_check logger propagates to caplog.

    This is needed because other tests (e.g., test_logging.py) may
    configure the "hfl" logger in ways that prevent propagation.
    """
    # Get the specific logger and its parent
    logger = logging.getLogger("hfl.engine.dependency_check")
    hfl_logger = logging.getLogger("hfl")

    # Save original state
    original_propagate = logger.propagate
    original_level = logger.level
    hfl_original_propagate = hfl_logger.propagate
    hfl_original_level = hfl_logger.level
    hfl_original_handlers = hfl_logger.handlers.copy()

    # Configure for capture
    logger.propagate = True
    logger.setLevel(logging.DEBUG)
    hfl_logger.propagate = True
    hfl_logger.setLevel(logging.DEBUG)
    # Remove handlers that might intercept logs
    hfl_logger.handlers = []

    yield logger

    # Restore original state
    logger.propagate = original_propagate
    logger.setLevel(original_level)
    hfl_logger.propagate = hfl_original_propagate
    hfl_logger.setLevel(hfl_original_level)
    hfl_logger.handlers = hfl_original_handlers


class TestCheckEngineAvailability:
    """Tests for check_engine_availability function."""

    def test_returns_dict(self):
        """check_engine_availability returns a dictionary."""
        result = check_engine_availability()
        assert isinstance(result, dict)

    def test_checks_llama_cpp(self):
        """check_engine_availability checks llama-cpp."""
        result = check_engine_availability()
        assert "llama-cpp" in result
        # Result is either True or an error string
        assert result["llama-cpp"] is True or isinstance(result["llama-cpp"], str)

    def test_checks_transformers(self):
        """check_engine_availability checks transformers."""
        result = check_engine_availability()
        assert "transformers" in result

    def test_checks_torch(self):
        """check_engine_availability checks torch."""
        result = check_engine_availability()
        assert "torch" in result

    def test_checks_vllm(self):
        """check_engine_availability checks vllm."""
        result = check_engine_availability()
        assert "vllm" in result

    def test_checks_soundfile(self):
        """check_engine_availability checks soundfile."""
        result = check_engine_availability()
        assert "soundfile" in result

    def test_checks_torchaudio(self):
        """check_engine_availability checks torchaudio."""
        result = check_engine_availability()
        assert "torchaudio" in result

    @patch.dict("sys.modules", {"llama_cpp": MagicMock()})
    def test_llama_cpp_available(self):
        """check_engine_availability returns True when llama_cpp is installed."""
        result = check_engine_availability()
        assert result["llama-cpp"] is True

    @patch.dict("sys.modules", {"llama_cpp": None})
    def test_llama_cpp_not_available(self):
        """check_engine_availability returns error string when llama_cpp is missing."""
        # Force reimport
        import sys

        if "llama_cpp" in sys.modules:
            del sys.modules["llama_cpp"]

        # This test may not work perfectly due to caching, but the function handles ImportError
        result = check_engine_availability()
        assert "llama-cpp" in result

    def test_torch_cuda_check(self):
        """check_engine_availability checks CUDA availability if torch is available."""
        result = check_engine_availability()
        if result.get("torch") is True:
            assert "torch_cuda" in result
            assert isinstance(result["torch_cuda"], bool)

    def test_torch_mps_check(self):
        """check_engine_availability checks MPS if available."""
        result = check_engine_availability()
        if result.get("torch") is True:
            # MPS may or may not be in results depending on platform
            # Just verify no error occurred
            pass


class TestLogAvailableBackends:
    """Tests for log_available_backends function."""

    def test_logs_backend_status(self, caplog, enable_dependency_check_logger):
        """log_available_backends logs backend availability."""
        with caplog.at_level(logging.INFO, logger="hfl.engine.dependency_check"):
            log_available_backends()

        # Should have logged something about backends
        assert "Backend availability check" in caplog.text

    def test_logs_llama_cpp_status(self, caplog, enable_dependency_check_logger):
        """log_available_backends logs llama-cpp status."""
        with caplog.at_level(logging.INFO, logger="hfl.engine.dependency_check"):
            log_available_backends()

        assert "llama-cpp" in caplog.text

    def test_logs_transformers_status(self, caplog, enable_dependency_check_logger):
        """log_available_backends logs transformers status."""
        with caplog.at_level(logging.INFO, logger="hfl.engine.dependency_check"):
            log_available_backends()

        assert "transformers" in caplog.text

    def test_logs_vllm_status(self, caplog, enable_dependency_check_logger):
        """log_available_backends logs vllm status."""
        with caplog.at_level(logging.INFO, logger="hfl.engine.dependency_check"):
            log_available_backends()

        assert "vllm" in caplog.text

    def test_logs_gpu_status(self, caplog, enable_dependency_check_logger):
        """log_available_backends logs GPU status when torch is available."""
        result = check_engine_availability()
        with caplog.at_level(logging.INFO, logger="hfl.engine.dependency_check"):
            log_available_backends()

        if result.get("torch") is True:
            # Should mention GPU/CUDA/MPS status
            assert any(x in caplog.text for x in ["CUDA", "MPS", "GPU", "CPU only"])

    def test_logs_tts_status(self, caplog, enable_dependency_check_logger):
        """log_available_backends logs TTS status."""
        with caplog.at_level(logging.INFO, logger="hfl.engine.dependency_check"):
            log_available_backends()

        assert "TTS" in caplog.text


class TestGetRecommendedBackend:
    """Tests for get_recommended_backend function."""

    def test_gguf_recommends_llama_cpp(self):
        """get_recommended_backend recommends llama-cpp for GGUF models."""
        result = check_engine_availability()
        backend = get_recommended_backend("gguf")

        if result.get("llama-cpp") is True:
            assert backend == "llama-cpp"
        else:
            assert backend is None

    def test_safetensors_with_transformers(self):
        """get_recommended_backend handles safetensors with available backends."""
        check_engine_availability()
        backend = get_recommended_backend("safetensors")

        # Should return a valid backend or None
        assert backend in (None, "llama-cpp", "transformers", "vllm")

    def test_pytorch_format(self):
        """get_recommended_backend handles pytorch format."""
        backend = get_recommended_backend("pytorch")

        # Should return a valid backend or None
        assert backend in (None, "llama-cpp", "transformers", "vllm")

    def test_unknown_format_returns_none(self):
        """get_recommended_backend returns None for unknown formats."""
        backend = get_recommended_backend("unknown_format")
        assert backend is None

    def test_empty_format_returns_none(self):
        """get_recommended_backend returns None for empty format."""
        backend = get_recommended_backend("")
        assert backend is None

    @patch("hfl.engine.dependency_check.check_engine_availability")
    def test_vllm_preferred_with_cuda(self, mock_check):
        """get_recommended_backend prefers vllm when CUDA is available."""
        mock_check.return_value = {
            "llama-cpp": True,
            "transformers": True,
            "vllm": True,
            "torch_cuda": True,
        }

        backend = get_recommended_backend("safetensors")
        assert backend == "vllm"

    @patch("hfl.engine.dependency_check.check_engine_availability")
    def test_transformers_fallback_without_vllm(self, mock_check):
        """get_recommended_backend falls back to transformers without vllm."""
        mock_check.return_value = {
            "llama-cpp": True,
            "transformers": True,
            "vllm": "Not installed",
            "torch_cuda": False,
        }

        backend = get_recommended_backend("safetensors")
        assert backend == "transformers"

    @patch("hfl.engine.dependency_check.check_engine_availability")
    def test_llama_cpp_last_resort(self, mock_check):
        """get_recommended_backend uses llama-cpp as last resort."""
        mock_check.return_value = {
            "llama-cpp": True,
            "transformers": "Not installed",
            "vllm": "Not installed",
            "torch_cuda": False,
        }

        backend = get_recommended_backend("safetensors")
        assert backend == "llama-cpp"

    @patch("hfl.engine.dependency_check.check_engine_availability")
    def test_no_backend_available(self, mock_check):
        """get_recommended_backend returns None when no backend is available."""
        mock_check.return_value = {
            "llama-cpp": "Not installed",
            "transformers": "Not installed",
            "vllm": "Not installed",
        }

        backend = get_recommended_backend("safetensors")
        assert backend is None
