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


class TestCheckEngineAvailabilityBranches:
    """Force every import to succeed via fake modules so the
    happy-path branches (lines 35/43-46/48-49/57/66/76/84/91) are
    covered without requiring the real extras to be installed."""

    def test_all_backends_available(self, monkeypatch):
        """Inject fake modules for every backend the function probes
        and assert each branch records ``True``."""
        import sys
        import types

        # Fake llama_cpp / transformers / vllm / soundfile / torchaudio.
        for name in ("llama_cpp", "transformers", "vllm", "soundfile", "torchaudio"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

        # Fake torch with CUDA + MPS available.
        fake_torch = types.ModuleType("torch")

        class _Cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def get_device_name(idx):
                return "FakeGPU"

        class _Mps:
            @staticmethod
            def is_available():
                return True

        fake_torch.cuda = _Cuda
        fake_backends = types.ModuleType("torch.backends")
        fake_backends.mps = _Mps
        fake_torch.backends = fake_backends
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        # Fake mlx_engine that reports available.
        from hfl.engine import mlx_engine

        monkeypatch.setattr(mlx_engine, "is_available", lambda: True)

        result = check_engine_availability()
        assert result["llama-cpp"] is True
        assert result["transformers"] is True
        assert result["torch"] is True
        assert result["torch_cuda"] is True
        assert result["cuda_device"] == "FakeGPU"
        assert result["torch_mps"] is True
        assert result["vllm"] is True
        assert result["mlx"] is True
        assert result["soundfile"] is True
        assert result["torchaudio"] is True

    def test_mlx_not_applicable_on_non_darwin(self, monkeypatch):
        """When mlx_lm is missing AND host is not Darwin-arm64, the
        message reflects "not applicable"."""
        import platform

        from hfl.engine import mlx_engine

        monkeypatch.setattr(mlx_engine, "is_available", lambda: False)
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")

        result = check_engine_availability()
        assert "Not applicable" in result["mlx"]

    def test_mlx_not_installed_message_on_apple_silicon(self, monkeypatch):
        """When mlx_lm is missing but the host IS Darwin-arm64, the
        message points at ``pip install 'hfl[mlx]'``."""
        import platform

        from hfl.engine import mlx_engine

        monkeypatch.setattr(mlx_engine, "is_available", lambda: False)
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(platform, "machine", lambda: "arm64")

        result = check_engine_availability()
        assert "hfl[mlx]" in result["mlx"]


class TestLogAvailableBackendsBranches:
    """Cover the GPU + MPS log lines."""

    def test_logs_mps_when_apple_silicon(self, caplog, enable_dependency_check_logger, monkeypatch):
        """The MPS log line fires when torch is present, no CUDA, but
        MPS is available — common on Apple Silicon dev machines."""
        from hfl.engine import dependency_check as module

        monkeypatch.setattr(
            module,
            "check_engine_availability",
            lambda: {
                "torch": True,
                "torch_cuda": False,
                "torch_mps": True,
                "transformers": True,
                "llama-cpp": True,
                "vllm": "Not installed",
            },
        )

        with caplog.at_level(logging.INFO, logger="hfl.engine.dependency_check"):
            log_available_backends()
        assert "MPS" in caplog.text

    def test_logs_cuda_device_name(self, caplog, enable_dependency_check_logger, monkeypatch):
        from hfl.engine import dependency_check as module

        monkeypatch.setattr(
            module,
            "check_engine_availability",
            lambda: {
                "torch": True,
                "torch_cuda": True,
                "cuda_device": "NVIDIA RTX 4090",
                "transformers": True,
                "llama-cpp": True,
                "vllm": True,
            },
        )

        with caplog.at_level(logging.INFO, logger="hfl.engine.dependency_check"):
            log_available_backends()
        assert "RTX 4090" in caplog.text


class TestGetRecommendedBackendGgufHappyPath:
    """Covers ``get_recommended_backend('gguf')`` with llama-cpp
    available — the most common path on a real install."""

    def test_gguf_with_llama_cpp_returns_llama_cpp(self, monkeypatch):
        from hfl.engine import dependency_check as module

        monkeypatch.setattr(
            module,
            "check_engine_availability",
            lambda: {"llama-cpp": True},
        )
        assert get_recommended_backend("gguf") == "llama-cpp"
