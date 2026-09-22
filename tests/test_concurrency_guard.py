# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the non-reentrant-backend concurrency clamp and the
configurable inference timeouts.

Both come from questions an integrator asked while deciding whether to
parallelise generation against a 72B:

- ``HFL_NUM_PARALLEL`` / ``OLLAMA_NUM_PARALLEL`` exist for drop-in parity with
  Ollama, where each parallel slot is a separate model *process*. HFL runs one
  in-process model instance, and llama.cpp keeps a single KV cache with **no
  internal lock** — so raising the knob used to let two threads interleave
  ``create_chat_completion`` on the same object. The failure mode is silently
  corrupted output, not an exception, which is why a warning is not enough and
  the dispatcher is clamped.
- ``generation_timeout`` was a hard-coded 600 s. A 70B at ~7 tok/s needs about
  five minutes for 2000 tokens *plus* prompt processing, so long-form
  generation legitimately exceeded it and surfaced as a 504 with no way to
  tune it short of editing the source.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest

from hfl.api.state import ServerState
from hfl.config import HFLConfig
from hfl.engine.dispatcher import InferenceDispatcher


def _engine(*, concurrent: bool, loaded: bool = True) -> MagicMock:
    """A stand-in engine declaring whether it tolerates concurrent inference."""
    engine = MagicMock()
    engine.supports_concurrent_inference = concurrent
    engine.is_loaded = loaded
    return engine


class TestDispatcherClamp:
    """``clamp_max_inflight`` only ever narrows, and only while drained."""

    def test_narrows_capacity_and_semaphore(self):
        d = InferenceDispatcher(max_inflight=4, max_queued=8)
        assert d.clamp_max_inflight(1) is True
        assert d.max_inflight == 1
        # The semaphore must actually be rebuilt, or the clamp is cosmetic and
        # two requests still run at once.
        assert d._sem._value == 1

    def test_never_widens(self):
        d = InferenceDispatcher(max_inflight=1, max_queued=8)
        assert d.clamp_max_inflight(8) is False
        assert d.max_inflight == 1

    def test_equal_limit_is_a_noop(self):
        d = InferenceDispatcher(max_inflight=2, max_queued=8)
        assert d.clamp_max_inflight(2) is False
        assert d.max_inflight == 2

    def test_rejects_zero_or_negative(self):
        d = InferenceDispatcher(max_inflight=4, max_queued=8)
        for bad in (0, -1):
            with pytest.raises(ValueError):
                d.clamp_max_inflight(bad)

    def test_refuses_while_a_slot_is_held(self):
        """Replacing the semaphore under an in-flight request would lose the
        outstanding permit. Refuse instead of corrupting the count."""
        d = InferenceDispatcher(max_inflight=4, max_queued=8)
        d._in_flight = 1
        assert d.clamp_max_inflight(1) is False
        assert d.max_inflight == 4

    @pytest.mark.asyncio
    async def test_clamped_dispatcher_really_serialises(self):
        """End-to-end: after the clamp, two coroutines cannot hold a slot at
        the same time."""
        d = InferenceDispatcher(max_inflight=4, max_queued=8, acquire_timeout=5)
        d.clamp_max_inflight(1)
        overlap = {"max": 0, "now": 0}

        async def worker():
            async with d.slot():
                overlap["now"] += 1
                overlap["max"] = max(overlap["max"], overlap["now"])
                await asyncio.sleep(0.02)
                overlap["now"] -= 1

        await asyncio.gather(*(worker() for _ in range(4)))
        assert overlap["max"] == 1


class TestEngineDeclaresConcurrency:
    def test_base_default_is_not_concurrent(self):
        """Every backend is assumed non-reentrant unless it says otherwise —
        the safe default, since getting it wrong corrupts output silently."""
        from hfl.engine.base import InferenceEngine

        assert InferenceEngine.supports_concurrent_inference.fget(object()) is False  # type: ignore[attr-defined]

    def test_llama_cpp_is_not_concurrent(self):
        from hfl.engine.llama_cpp import LlamaCppEngine

        assert LlamaCppEngine().supports_concurrent_inference is False

    def test_vllm_is_concurrent(self):
        """vLLM batches internally; it is the one backend where the knob is
        meaningful."""
        from hfl.engine.vllm_engine import VLLMEngine

        assert VLLMEngine.supports_concurrent_inference.fget(object()) is True  # type: ignore[attr-defined]


class TestStateEnforcesConcurrency:
    """The clamp is applied when the engine is loaded, because the backend is
    unknown when the dispatcher is built."""

    @pytest.mark.asyncio
    async def test_non_reentrant_engine_clamps_to_one(self, caplog):
        state = ServerState()
        dispatcher = InferenceDispatcher(max_inflight=4, max_queued=8)
        with patch.object(ServerState, "_try_get_dispatcher", return_value=dispatcher):
            with caplog.at_level(logging.WARNING, logger="hfl.api.state"):
                await state.set_llm_engine(_engine(concurrent=False), MagicMock())

        assert dispatcher.max_inflight == 1
        assert "clamped to 1" in caplog.text
        # The message must name the knob, or the operator has no idea why
        # their setting was ignored.
        assert "NUM_PARALLEL" in caplog.text

    @pytest.mark.asyncio
    async def test_concurrent_engine_keeps_operator_setting(self):
        state = ServerState()
        dispatcher = InferenceDispatcher(max_inflight=4, max_queued=8)
        with patch.object(ServerState, "_try_get_dispatcher", return_value=dispatcher):
            await state.set_llm_engine(_engine(concurrent=True), MagicMock())

        assert dispatcher.max_inflight == 4

    @pytest.mark.asyncio
    async def test_clamp_applies_on_swap_too(self):
        """A swap from vLLM to llama.cpp must re-clamp; the drained
        ``exclusive()`` window is what makes it safe."""
        state = ServerState()
        dispatcher = InferenceDispatcher(max_inflight=4, max_queued=8)
        with patch.object(ServerState, "_try_get_dispatcher", return_value=dispatcher):
            await state.set_llm_engine(_engine(concurrent=True), MagicMock())
            assert dispatcher.max_inflight == 4
            with patch("asyncio.to_thread", new=MagicMock(return_value=asyncio.sleep(0))):
                await state.set_llm_engine(_engine(concurrent=False), MagicMock())

        assert dispatcher.max_inflight == 1

    @pytest.mark.asyncio
    async def test_unloading_to_none_does_not_clamp(self):
        state = ServerState()
        dispatcher = InferenceDispatcher(max_inflight=4, max_queued=8)
        with patch.object(ServerState, "_try_get_dispatcher", return_value=dispatcher):
            await state.set_llm_engine(None, None)
        assert dispatcher.max_inflight == 4

    @pytest.mark.asyncio
    async def test_missing_dispatcher_is_not_an_error(self):
        """Unit tests that never built the container must still be able to set
        an engine."""
        state = ServerState()
        with patch.object(ServerState, "_try_get_dispatcher", return_value=None):
            await state.set_llm_engine(_engine(concurrent=False), MagicMock())
        assert state.engine is not None

    @pytest.mark.asyncio
    async def test_engine_without_the_property_is_treated_as_unsafe(self):
        """A third-party engine predating the property must not be assumed
        concurrent — ``getattr(..., False)`` is the fail-safe direction."""
        state = ServerState()
        dispatcher = InferenceDispatcher(max_inflight=4, max_queued=8)
        engine = MagicMock(spec=["is_loaded", "unload"])
        engine.is_loaded = True
        with patch.object(ServerState, "_try_get_dispatcher", return_value=dispatcher):
            await state.set_llm_engine(engine, MagicMock())
        assert dispatcher.max_inflight == 1


class TestConfigurableTimeouts:
    def test_generation_timeout_defaults_to_600(self):
        assert HFLConfig().generation_timeout == 600.0

    def test_model_load_timeout_defaults_to_300(self):
        assert HFLConfig().model_load_timeout == 300.0

    def test_generation_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("HFL_GENERATION_TIMEOUT", "1800")
        assert HFLConfig().generation_timeout == 1800.0

    def test_model_load_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("HFL_MODEL_LOAD_TIMEOUT", "900")
        assert HFLConfig().model_load_timeout == 900.0

    def test_ollama_load_timeout_fallback(self, monkeypatch):
        """Drop-in parity: an existing Ollama deployment already sets this."""
        monkeypatch.setenv("OLLAMA_LOAD_TIMEOUT", "1200")
        assert HFLConfig().model_load_timeout == 1200.0

    def test_hfl_wins_over_ollama(self, monkeypatch):
        monkeypatch.setenv("HFL_MODEL_LOAD_TIMEOUT", "111")
        monkeypatch.setenv("OLLAMA_LOAD_TIMEOUT", "999")
        assert HFLConfig().model_load_timeout == 111.0

    def test_fractional_values_survive(self, monkeypatch):
        monkeypatch.setenv("HFL_GENERATION_TIMEOUT", "0.5")
        assert HFLConfig().generation_timeout == 0.5

    @pytest.mark.asyncio
    async def test_run_dispatched_honours_the_configured_timeout(self, monkeypatch):
        """The knob has to reach the code path that enforces it, not just the
        config object."""
        from fastapi import HTTPException

        import hfl.api.helpers as helpers

        monkeypatch.setattr(helpers.config, "generation_timeout", 0.05)
        dispatcher = InferenceDispatcher(max_inflight=1, max_queued=4, acquire_timeout=5)
        monkeypatch.setattr("hfl.core.get_dispatcher", lambda: dispatcher, raising=False)

        def _slow() -> str:
            import time

            time.sleep(0.5)
            return "done"

        with patch("hfl.core.get_dispatcher", return_value=dispatcher):
            with pytest.raises(HTTPException) as exc:
                await helpers.run_dispatched(_slow, operation="chat")

        assert exc.value.status_code == 504
        assert exc.value.detail["timeout_seconds"] == 0.05
