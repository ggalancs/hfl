# SPDX-License-Identifier: Apache-2.0
"""Edge case tests for engine selector module."""

import pytest

from hfl.engine.selector import MissingDependencyError


class TestMissingDependencyErrorDetails:
    """Tests for MissingDependencyError details."""

    def test_error_message_multiline(self):
        """Test error message with multiple lines."""
        error = MissingDependencyError(
            "The vLLM backend requires additional dependencies.\n\n"
            "Install them with:\n"
            "  pip install hfl[vllm]"
        )

        message = str(error)
        assert "pip install" in message
        assert "vllm" in message.lower()

    def test_error_can_be_raised_and_caught(self):
        """Test error can be raised and caught properly."""
        with pytest.raises(MissingDependencyError) as exc_info:
            raise MissingDependencyError("Test error")

        assert "Test error" in str(exc_info.value)

    def test_error_is_exception_subclass(self):
        """Test error is properly subclassed from Exception."""
        error = MissingDependencyError("Test")

        assert isinstance(error, Exception)
        assert isinstance(error, MissingDependencyError)

    def test_error_with_long_message(self):
        """Test error with installation instructions."""
        error = MissingDependencyError(
            "The llama-cpp backend requires the 'llama-cpp-python' library.\n\n"
            "Install it with:\n"
            "  pip install llama-cpp-python\n\n"
            "For GPU support (CUDA):\n"
            '  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python\n\n'
            "For macOS with Metal:\n"
            '  CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python'
        )

        message = str(error)
        assert "llama-cpp-python" in message
        assert "CUDA" in message
        assert "Metal" in message


def test_a_llama_cpp_whose_native_library_does_not_load_is_a_clear_501(monkeypatch) -> None:
    """Installed but unloadable (an executable that did not bundle libllama:
    FileNotFoundError on import) was an opaque 500; it says what failed."""
    import sys
    import types

    from hfl.engine.selector import MissingDependencyError, _get_llama_cpp_engine

    broken = types.ModuleType("hfl.engine.llama_cpp")

    def missing_library(name: str):
        raise FileNotFoundError("Shared library with base name 'llama' not found")

    broken.__getattr__ = missing_library  # type: ignore[method-assign]
    monkeypatch.setitem(sys.modules, "hfl.engine.llama_cpp", broken)
    with pytest.raises(MissingDependencyError) as caught:
        _get_llama_cpp_engine()
    assert caught.value.status_code == 501
    assert "native library did not load" in str(caught.value)
    assert "Shared library with base name 'llama' not found" in str(caught.value)
