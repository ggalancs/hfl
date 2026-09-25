# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Global pytest configuration and shared fixtures.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Every test runs against a throwaway HFL home. Set before any test module
# imports hfl, so the global config is built from it; a test that forgets
# ``temp_config`` then writes here instead of the developer's ~/.hfl.
_SESSION_HOME = tempfile.TemporaryDirectory(prefix="hfl-test-home-")
os.environ["HFL_HOME"] = _SESSION_HOME.name
# Nor does it see a llama-server the developer happens to have installed:
# without llama-cpp-python, GGUF models fall back to it, and the suite would
# start real processes against its fake GGUF files. Tests that need one point
# HFL_LLAMA_SERVER_BIN at their own.
os.environ["HFL_LLAMA_SERVER_BIN"] = os.path.join(_SESSION_HOME.name, "no-llama-server")


def pytest_unconfigure(config):
    _SESSION_HOME.cleanup()


@pytest.fixture(autouse=True)
def set_english_language(monkeypatch):
    """Ensure all tests run with English language and no colors."""
    monkeypatch.setenv("HFL_LANG", "en")
    # Disable Rich color output to avoid ANSI escape codes in tests
    monkeypatch.setenv("NO_COLOR", "1")
    # Clear the language cache so it picks up the new env var
    from hfl.i18n import get_language

    get_language.cache_clear()


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter storage before each test to prevent 429 errors.

    Resets the middleware instance's request counts AND walks the app's
    middleware stack to reset any instance that may differ from the global
    reference (e.g., after middleware stack rebuild).
    """
    try:
        import hfl.api.server  # noqa: F401 - ensures app is created
        from hfl.api.middleware import RateLimitMiddleware
        from hfl.api.middleware import reset_rate_limiter as do_reset

        def _reset_all() -> None:
            # Reset the global-tracked instance
            do_reset()
            # Also walk the middleware stack to find and reset any instance
            # that might differ from _rate_limiter_instance
            app = hfl.api.server.app
            current = getattr(app, "middleware_stack", None)
            while current is not None:
                if isinstance(current, RateLimitMiddleware):
                    current.reset()
                current = getattr(current, "app", None)

        _reset_all()
        yield
        _reset_all()
    except ImportError:
        # If server module isn't available, just yield
        yield


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

    import hfl.hub.blobs

    monkeypatch.setattr(hfl.hub.blobs, "config", test_config)

    # Reset registry singleton cache so it uses the new config
    from hfl.models.registry import reset_registry

    reset_registry()

    yield test_config

    # Clean up: reset registry after test
    reset_registry()


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

    registry.add(
        ModelManifest(
            name="another-model-q5_k_m",
            repo_id="other-org/another-model",
            local_path="/tmp/another-model",
            format="gguf",
            size_bytes=10 * 1024**3,
            quantization="Q5_K_M",
        )
    )

    return registry


@pytest.fixture
def mock_llama_cpp():
    """Mock of the complete llama_cpp module."""
    mock_llama = MagicMock()
    mock_llama_class = MagicMock()
    mock_llama.Llama = mock_llama_class

    with patch.dict(sys.modules, {"llama_cpp": mock_llama}):
        yield mock_llama_class


# ----------------------------------------------------------------------
# Global-state pollution guard
# ----------------------------------------------------------------------
#
# Several tests swap ``sys.modules["llama_cpp"]`` for a stub to exercise
# the "backend missing / too old" paths. Two of them used to put the stub
# in and never take it out, and the damage was invisible for a long time:
# a test that merely checks the backend is importable passes happily
# against a stub, so it reports green while testing nothing. It only
# surfaced when a test introspected the real package and got an
# AttributeError — two files and several hundred tests later.
#
# This hook names the test that left the stub behind, at the moment it
# happens, instead of leaving the next person to bisect for it.


def _llama_cpp_is_stubbed() -> bool:
    module = sys.modules.get("llama_cpp")
    if module is None:
        return False
    # The real package carries the ctypes bindings; every stub in this
    # suite is a bare ModuleType or a MagicMock standing in for it.
    return not hasattr(module, "llama_model_params")


@pytest.fixture(autouse=True)
def _no_torch_pollution(request):
    """Same guard for ``torch``: a MagicMock left in ``sys.modules`` made
    ``pytest.importorskip("torch")`` succeed without torch installed, and a
    later test compute on mocks (test_selector's CUDA test did this)."""
    before = sys.modules.get("torch")
    yield
    after = sys.modules.get("torch")
    if after is not before and after is not None and getattr(after, "__file__", None) is None:
        raise AssertionError(
            f"{request.node.nodeid} left a stub torch in sys.modules; "
            "monkeypatch.setitem(sys.modules, 'torch', ...) restores it for you."
        )


@pytest.fixture(autouse=True)
def _no_llama_cpp_pollution(request):
    """Fail the test that leaves a stub behind, by name.

    An autouse fixture declared in the root conftest is the OUTERMOST
    one, so its teardown runs last — after ``monkeypatch`` has already
    undone a well-behaved test's swap. That ordering is the whole reason
    it is a fixture rather than a ``pytest_runtest_protocol`` wrapper:
    failing from inside that hook produces an INTERNALERROR and still
    reports the test as passed, which is precisely the kind of
    unattributable result this guard exists to prevent.
    """
    polluted_before = _llama_cpp_is_stubbed()
    yield
    if _llama_cpp_is_stubbed() and not polluted_before:
        raise AssertionError(
            f"{request.node.nodeid} left a stub llama_cpp in sys.modules. "
            "Restore it — monkeypatch.setitem swaps the entry back for you. Do "
            "NOT delete and re-import: a native extension re-initialised "
            "mid-process crashes the interpreter. Left in place, the stub "
            "silently shadows the real package for every test that follows, so "
            "anything merely checking the backend is importable reports green "
            "while testing the stub."
        )
