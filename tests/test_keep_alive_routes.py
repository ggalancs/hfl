# SPDX-License-Identifier: Apache-2.0
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

    def test_an_explicit_keep_alive_sticks_and_restarts_on_use(self, client, sample_manifest):
        """A later request that omits the field keeps the model's explicit
        keep_alive, and — since keep_alive is measured from the last use —
        restarts its clock. The old rule kept the first absolute deadline,
        which once enforced would unload a model in continuous use."""
        _mock_llm_loaded(sample_manifest)
        first = client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "hi",
                "stream": False,
                "keep_alive": "1h",
            },
        )
        assert first.status_code == 200

        before = datetime.now(timezone.utc)
        response = client.post(
            "/api/generate",
            json={"model": sample_manifest.name, "prompt": "hi", "stream": False},
        )
        assert response.status_code == 200
        deadline = get_state().keep_alive_deadline_for(sample_manifest.name)
        assert deadline is not None
        assert abs((deadline - (before + timedelta(hours=1))).total_seconds()) < 30


class TestKeepAliveGlobalDefault:
    """``HFL_KEEP_ALIVE`` (and the Ollama alias) installs a server-wide
    default for requests that don't carry their own ``keep_alive``."""

    def test_global_default_applied_when_request_omits_keep_alive(self, client, sample_manifest):
        from hfl.config import config as hfl_config

        _mock_llm_loaded(sample_manifest)
        # A preloaded model already follows the default from load time; the
        # request must restart it with the default in force NOW (10m).

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

    def test_global_default_does_not_overwrite_an_explicit_value(self, client, sample_manifest):
        """Explicit decisions win: a model given keep_alive=2h keeps 2h on a
        follow-up request that omits the field, not the 5m default."""
        from hfl.config import config as hfl_config

        _mock_llm_loaded(sample_manifest)
        previous = hfl_config.keep_alive_default
        hfl_config.keep_alive_default = "5m"
        try:
            client.post(
                "/api/generate",
                json={
                    "model": sample_manifest.name,
                    "prompt": "hi",
                    "stream": False,
                    "keep_alive": "2h",
                },
            )
            before = datetime.now(timezone.utc)
            response = client.post(
                "/api/generate",
                json={"model": sample_manifest.name, "prompt": "hi", "stream": False},
            )
            assert response.status_code == 200
            deadline = get_state().keep_alive_deadline_for(sample_manifest.name)
            assert abs((deadline - (before + timedelta(hours=2))).total_seconds()) < 30
        finally:
            hfl_config.keep_alive_default = previous

    def test_invalid_global_default_falls_back_to_five_minutes(self, client, sample_manifest):
        """A misconfigured server must not break user requests — nor pin a
        model's memory forever on a typo, now that keep_alive is enforced."""
        from hfl.config import config as hfl_config

        _mock_llm_loaded(sample_manifest)
        previous = hfl_config.keep_alive_default
        hfl_config.keep_alive_default = "not-a-duration"
        try:
            before = datetime.now(timezone.utc)
            response = client.post(
                "/api/generate",
                json={"model": sample_manifest.name, "prompt": "hi", "stream": False},
            )
            assert response.status_code == 200
            deadline = get_state().keep_alive_deadline_for(sample_manifest.name)
            assert abs((deadline - (before + timedelta(minutes=5))).total_seconds()) < 30
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
        """``keep_alive=0`` should schedule the eviction of THAT model as a
        FastAPI BackgroundTask, after the response is flushed — and only
        that model: others stay resident. We verify by intercepting
        ``evict``.
        """
        _mock_llm_loaded(sample_manifest)
        evicted: list[str] = []

        async def fake_evict(name, reason="requested"):
            evicted.append(name)
            return True

        state = get_state()
        original = state.evict
        state.evict = AsyncMock(side_effect=fake_evict)  # type: ignore[assignment]
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
            state.evict = original  # type: ignore[method-assign]

        assert response.status_code == 200
        # TestClient runs background tasks synchronously after the
        # response is built, so by the time .json() returns the unload
        # has completed.
        assert evicted == [sample_manifest.name]


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
