# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Integration tests for ``keep_alive`` on the Ollama-native routes.

Exercises the end-to-end contract: the field is accepted, parsed,
rejected when malformed, and translated into either a deadline on the
server state (which ``/api/ps`` surfaces) or a background unload task.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


def _mock_llm_loaded(sample_manifest):
    """Wire up state so ``_ensure_model_loaded`` returns instantly."""
    state = get_state()
    engine = MagicMock()
    engine.is_loaded = True
    engine.chat = MagicMock(
        return_value=MagicMock(text="ok", tokens_generated=1, tokens_prompt=1, stop_reason="stop")
    )
    engine.generate = MagicMock(
        return_value=MagicMock(text="ok", tokens_generated=1, tokens_prompt=1, stop_reason="stop")
    )
    state.engine = engine
    state.current_model = sample_manifest


class TestKeepAliveDeadlineOnGenerate:
    def test_keep_alive_5m_records_future_deadline(self, client, sample_manifest):
        _mock_llm_loaded(sample_manifest)
        before = datetime.now(timezone.utc)

        response = client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "hi",
                "stream": False,
                "keep_alive": "5m",
            },
        )
        assert response.status_code == 200

        deadline = get_state().keep_alive_deadline_for(sample_manifest.name)
        assert deadline is not None
        # Deadline is in the future, roughly "now + 5 minutes".
        delta = deadline - before
        assert timedelta(minutes=4, seconds=55) <= delta <= timedelta(minutes=5, seconds=5)

    def test_keep_alive_numeric_seconds(self, client, sample_manifest):
        _mock_llm_loaded(sample_manifest)
        before = datetime.now(timezone.utc)
        response = client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "hi",
                "stream": False,
                "keep_alive": 10,
            },
        )
        assert response.status_code == 200

        deadline = get_state().keep_alive_deadline_for(sample_manifest.name)
        assert deadline is not None
        assert timedelta(seconds=9) <= deadline - before <= timedelta(seconds=11)

    def test_keep_alive_minus_one_clears_deadline(self, client, sample_manifest):
        _mock_llm_loaded(sample_manifest)
        # Pre-populate with some deadline
        get_state().set_keep_alive_deadline(
            sample_manifest.name, datetime.now(timezone.utc) + timedelta(hours=2)
        )

        response = client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "hi",
                "stream": False,
                "keep_alive": -1,
            },
        )
        assert response.status_code == 200

        # "-1" → never expire → deadline cleared (null in /api/ps)
        assert get_state().keep_alive_deadline_for(sample_manifest.name) is None

    def test_missing_keep_alive_leaves_deadline_untouched(self, client, sample_manifest):
        _mock_llm_loaded(sample_manifest)
        existing = datetime.now(timezone.utc) + timedelta(hours=1)
        get_state().set_keep_alive_deadline(sample_manifest.name, existing)

        response = client.post(
            "/api/generate",
            json={"model": sample_manifest.name, "prompt": "hi", "stream": False},
        )
        assert response.status_code == 200

        # No keep_alive in body → previous deadline survives unchanged
        assert get_state().keep_alive_deadline_for(sample_manifest.name) == existing


class TestKeepAliveGlobalDefault:
    """``HFL_KEEP_ALIVE`` (and the Ollama alias) installs a server-wide
    default for requests that don't carry their own ``keep_alive``."""

    def test_global_default_applied_when_request_omits_keep_alive(
        self, client, sample_manifest
    ):
        from hfl.config import config as hfl_config

        _mock_llm_loaded(sample_manifest)
        # No deadline yet → global default kicks in.
        assert get_state().keep_alive_deadline_for(sample_manifest.name) is None

        previous = hfl_config.keep_alive_default
        hfl_config.keep_alive_default = "10m"
        try:
            before = datetime.now(timezone.utc)
            response = client.post(
                "/api/generate",
                json={"model": sample_manifest.name, "prompt": "hi", "stream": False},
            )
            assert response.status_code == 200
            deadline = get_state().keep_alive_deadline_for(sample_manifest.name)
            assert deadline is not None
            # 10 minutes ± a few seconds slack for the round-trip.
            expected = before + timedelta(minutes=10)
            assert abs((deadline - expected).total_seconds()) < 30
        finally:
            hfl_config.keep_alive_default = previous

    def test_global_default_does_not_overwrite_existing_deadline(
        self, client, sample_manifest
    ):
        """Once a previous request set a deadline, omitting ``keep_alive``
        on a follow-up request must NOT reset it to the global default —
        explicit decisions win."""
        from hfl.config import config as hfl_config

        _mock_llm_loaded(sample_manifest)
        sticky = datetime.now(timezone.utc) + timedelta(hours=2)
        get_state().set_keep_alive_deadline(sample_manifest.name, sticky)

        previous = hfl_config.keep_alive_default
        hfl_config.keep_alive_default = "5m"
        try:
            response = client.post(
                "/api/generate",
                json={"model": sample_manifest.name, "prompt": "hi", "stream": False},
            )
            assert response.status_code == 200
            assert get_state().keep_alive_deadline_for(sample_manifest.name) == sticky
        finally:
            hfl_config.keep_alive_default = previous

    def test_invalid_global_default_is_ignored_silently(self, client, sample_manifest):
        """A misconfigured server must not break user requests."""
        from hfl.config import config as hfl_config

        _mock_llm_loaded(sample_manifest)
        previous = hfl_config.keep_alive_default
        hfl_config.keep_alive_default = "not-a-duration"
        try:
            response = client.post(
                "/api/generate",
                json={"model": sample_manifest.name, "prompt": "hi", "stream": False},
            )
            assert response.status_code == 200
            assert get_state().keep_alive_deadline_for(sample_manifest.name) is None
        finally:
            hfl_config.keep_alive_default = previous


class TestKeepAliveDeadlineOnChat:
    def test_chat_respects_keep_alive(self, client, sample_manifest):
        _mock_llm_loaded(sample_manifest)
        response = client.post(
            "/api/chat",
            json={
                "model": sample_manifest.name,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "keep_alive": "30s",
            },
        )
        assert response.status_code == 200

        deadline = get_state().keep_alive_deadline_for(sample_manifest.name)
        assert deadline is not None


class TestKeepAliveZeroTriggersUnload:
    def test_keep_alive_zero_schedules_background_unload(self, client, sample_manifest):
        """``keep_alive=0`` should schedule ``state.cleanup`` as a
        FastAPI BackgroundTask so the model is evicted after the
        response is flushed. We verify by intercepting ``cleanup``.
        """
        _mock_llm_loaded(sample_manifest)
        cleanup_called = False

        async def fake_cleanup():
            nonlocal cleanup_called
            cleanup_called = True

        state = get_state()
        # Patch the async method ``cleanup`` on the specific instance.
        original = state.cleanup
        state.cleanup = AsyncMock(side_effect=fake_cleanup)  # type: ignore[assignment]
        try:
            response = client.post(
                "/api/generate",
                json={
                    "model": sample_manifest.name,
                    "prompt": "hi",
                    "stream": False,
                    "keep_alive": 0,
                },
            )
        finally:
            state.cleanup = original  # restore

        assert response.status_code == 200
        # TestClient runs background tasks synchronously after the
        # response is built, so by the time .json() returns the unload
        # has completed.
        assert cleanup_called


class TestKeepAliveRejectsInvalid:
    @pytest.mark.parametrize("bad", ["5minutes", "abc", "-2m", "1d", []])
    def test_invalid_keep_alive_returns_400(self, client, sample_manifest, bad):
        _mock_llm_loaded(sample_manifest)
        response = client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "hi",
                "stream": False,
                "keep_alive": bad,
            },
        )
        # ``list`` is rejected by Pydantic with 422; strings land in
        # our domain validator at 400.
        assert response.status_code in (400, 422)


class TestKeepAliveSurfacedOnPs:
    def test_ps_reflects_deadline_set_by_keep_alive(self, client, sample_manifest):
        """End-to-end: /api/generate with ``keep_alive=10m`` →
        /api/ps shows a non-null ``expires_at``."""
        _mock_llm_loaded(sample_manifest)
        client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "hi",
                "stream": False,
                "keep_alive": "10m",
            },
        )

        entry = client.get("/api/ps").json()["models"][0]
        assert entry["expires_at"] is not None
        assert entry["expires_at"].endswith("Z")
