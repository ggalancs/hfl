# SPDX-License-Identifier: HRUL-1.0
"""Tests for API model loading and switching."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state


class TestModelSwitching:
    """Tests for model switching in API."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset server state before each test."""
        get_state().api_key = None
        get_state().engine = None
        get_state().current_model = None
        yield
        get_state().api_key = None
        get_state().engine = None
        get_state().current_model = None

    def test_load_different_model_unloads_current(self):
        """Test that loading a different model unloads the current one."""
        # Setup: pretend we have a model loaded
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.unload = MagicMock()

        mock_current_model = MagicMock()
        mock_current_model.name = "model-a"

        get_state().engine = mock_engine
        get_state().current_model = mock_current_model

        client = TestClient(app)

        # Try to use a different model - this will fail but should trigger unload
        with patch("hfl.api.model_loader.get_registry") as mock_get_registry:
            mock_registry = MagicMock()
            mock_registry.get.return_value = None  # Model not found
            mock_get_registry.return_value = mock_registry

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "model-b",  # Different model
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

            # Should return 404 for model not found
            assert response.status_code == 404

    def test_same_model_no_reload(self):
        """Test that requesting the same model doesn't reload it."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True

        mock_current_model = MagicMock()
        mock_current_model.name = "test-model"

        get_state().engine = mock_engine
        get_state().current_model = mock_current_model

        client = TestClient(app)

        # Mock the chat method to return a result
        mock_engine.chat.return_value = MagicMock(
            text="Hello!",
            tokens_generated=5,
            tokens_prompt=3,
            tokens_per_second=50.0,
            stop_reason="stop",
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",  # Same model
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        # Should succeed without reloading
        assert response.status_code == 200
        # Engine should not be unloaded
        mock_engine.unload.assert_not_called()


class TestModelNotFound:
    """Tests for model not found scenarios."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset server state before each test."""
        get_state().api_key = None
        get_state().engine = None
        get_state().current_model = None
        yield

    def test_chat_completions_model_not_found(self):
        """Test chat completions with non-existent model."""
        client = TestClient(app)

        with patch("hfl.api.model_loader.get_registry") as mock_get_registry:
            mock_registry = MagicMock()
            mock_registry.get.return_value = None
            mock_get_registry.return_value = mock_registry

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "nonexistent-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

            assert response.status_code == 404
            # Envelope after R10: ``{"error": "...", "code":
            # "ModelNotFoundError", ...}``; ``detail`` kept as a
            # legacy-reader fallback.
            body = response.json()
            msg = body.get("error") or body.get("detail") or ""
            assert "not found" in str(msg).lower()

    def test_completions_model_not_found(self):
        """Test completions with non-existent model."""
        client = TestClient(app)

        with patch("hfl.api.model_loader.get_registry") as mock_get_registry:
            mock_registry = MagicMock()
            mock_registry.get.return_value = None
            mock_get_registry.return_value = mock_registry

            response = client.post(
                "/v1/completions", json={"model": "nonexistent-model", "prompt": "Hello"}
            )

            assert response.status_code == 404


class TestAPIVersion:
    """Tests for API version endpoint."""

    def test_api_version(self):
        """Test /api/version endpoint."""
        client = TestClient(app)

        response = client.get("/api/version")

        assert response.status_code == 200
        assert "version" in response.json()


class TestColdLoadCoalescing:
    """CON: concurrent first-requests for the SAME cold model must coalesce into
    a single engine.load() instead of each running a multi-GB load (2x VRAM /
    OOM / A-B thrash). load_llm now routes through ensure_llm_loaded."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        from hfl.api.state import reset_state

        reset_state()
        yield
        reset_state()

    @pytest.mark.asyncio
    async def test_concurrent_cold_loads_run_loader_once(self):
        import asyncio
        import threading
        import time

        from hfl.api import model_loader
        from hfl.converter.formats import ModelType
        from hfl.core import get_dispatcher

        get_dispatcher().reset()
        try:
            load_calls = 0
            guard = threading.Lock()

            def _load(path, **kwargs):
                nonlocal load_calls
                with guard:
                    load_calls += 1
                time.sleep(0.05)  # widen the race window

            engine = MagicMock()
            engine.is_loaded = True
            engine.load = MagicMock(side_effect=_load)

            manifest = MagicMock()
            manifest.name = "cold-model"
            manifest.local_path = "/fake/cold.gguf"

            with (
                patch("hfl.api.model_loader.get_registry") as mock_reg,
                patch("hfl.api.model_loader.detect_model_type", return_value=ModelType.LLM),
                patch("hfl.api.model_loader.select_engine", return_value=engine),
            ):
                mock_reg.return_value.get.return_value = manifest
                results = await asyncio.gather(
                    *[model_loader.load_llm("cold-model") for _ in range(6)]
                )

            assert all(r[0] is engine for r in results)
            assert load_calls == 1, f"engine.load ran {load_calls} times (expected 1)"
        finally:
            get_dispatcher().reset()
