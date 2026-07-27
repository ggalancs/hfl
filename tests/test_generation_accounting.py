# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for token accounting and the context-reload decision.

Both come from the same incident. A client reported "4 tok/s instead of 9"
against a live 72B, and the server could not answer the question:

- ``hfl_tokens_generated_total`` read ``0`` forever. The counter, its
  Prometheus export and the event listener that feeds it all existed; the
  only emitter of ``GENERATION_COMPLETED`` sits in ``EngineObserver``, which
  is wired to nothing. Accounting now happens in ``run_dispatched``.
- Nothing logged the prefill/generation split, so a 61 s request could have
  been 550 tokens at 9 tok/s or 264 tokens after 32 s of prompt processing.
  A client computing tokens ÷ wall-clock sees ~4 tok/s in the second case
  while generation is running at full speed.

And the log showed the model reloading 16 times across 86 requests because
the client varied ``num_ctx`` between 8192/16384/32768. Six of those reloads
*shrank* the window — pure waste, since a model opened at 32768 already
serves an 8192-token request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hfl.api.helpers import _account_generation
from hfl.api.model_loader import load_llm
from hfl.api.state import reset_state


@dataclass
class _Result:
    """Stand-in for the engine's GenerationResult."""

    tokens_generated: int = 264
    tokens_prompt: int = 3021
    prompt_eval_duration: int = 32_400_000_000  # 32.4 s in ns
    eval_duration: int = 29_000_000_000  # 29.0 s in ns
    total_duration: int = 61_400_000_000  # 61.4 s in ns


class TestAccounting:
    def test_counters_are_incremented(self):
        """The regression: these stayed at 0 on a server that had served 65
        inferences."""
        metrics = MagicMock()
        with patch("hfl.metrics.get_metrics", return_value=metrics):
            _account_generation(_Result(), "chat")

        metrics.record_generation.assert_called_once()
        kwargs = metrics.record_generation.call_args.kwargs
        assert kwargs["tokens_in"] == 3021
        assert kwargs["tokens_out"] == 264
        assert kwargs["duration_ms"] == pytest.approx(61_400, rel=1e-3)

    def test_log_separates_prefill_from_generation(self, caplog):
        """The whole point: 264 tokens over 61.4 s wall-clock is ~4 tok/s, but
        generation itself ran at 9.1. Both numbers must be visible."""
        with caplog.at_level(logging.INFO, logger="hfl.api.helpers"):
            with patch("hfl.metrics.get_metrics", return_value=MagicMock()):
                _account_generation(_Result(), "chat")

        msg = caplog.text
        assert "prompt 3021 tok" in msg
        assert "93.2 tok/s" in msg  # 3021 / 32.4 s — prefill
        assert "generated 264 tok" in msg
        assert "9.1 tok/s" in msg  # 264 / 29.0 s — real generation speed
        # And the misleading wall-clock rate (264 / 61.4 = 4.3) is NOT what
        # the line reports as the generation speed.
        assert "4.3 tok/s" not in msg

    def test_non_generation_results_are_ignored(self):
        """``run_dispatched`` also carries embeddings and TTS, whose results
        have none of these fields."""
        metrics = MagicMock()
        with patch("hfl.metrics.get_metrics", return_value=metrics):
            _account_generation([0.1, 0.2, 0.3], "embed")
            _account_generation(None, "tts")
            _account_generation(MagicMock(spec=[]), "other")
        metrics.record_generation.assert_not_called()

    def test_missing_timings_do_not_crash(self, caplog):
        """Engines that don't measure leave the durations at 0; the line must
        degrade to n/a rather than divide by zero."""
        with caplog.at_level(logging.INFO, logger="hfl.api.helpers"):
            with patch("hfl.metrics.get_metrics", return_value=MagicMock()):
                _account_generation(
                    _Result(prompt_eval_duration=0, eval_duration=0, total_duration=0),
                    "chat",
                )
        assert "n/a" in caplog.text

    def test_metrics_failure_never_breaks_the_request(self):
        """Accounting is diagnostics; a broken metrics backend must not turn a
        successful generation into a 500."""
        with patch("hfl.metrics.get_metrics", side_effect=RuntimeError("boom")):
            _account_generation(_Result(), "chat")  # must not raise


class TestDurationSplit:
    """``prompt_eval_duration`` / ``eval_duration`` must be *measured*.

    They used to be derived by splitting total wall-clock in proportion to
    token counts, which silently assumes prefill and generation run at the
    same tokens/second. They differ by roughly an order of magnitude. On a
    real 72B request — 4161-token prompt, 250 generated, 99 s total — the
    proportional split handed 94 % of the time to the prompt and reported
    generation at 44.8 tok/s, on hardware whose ceiling is about 9 tok/s.
    Reading llama.cpp's own counters gave 69 tok/s prefill and 7.2 tok/s
    generation, which is physically coherent.
    """

    def test_uses_measured_counters_when_available(self):
        from hfl.engine import llama_cpp as m

        # 60.1 s of prefill, 7.2 s of generation — the measured numbers.
        perf = (60_100_000_000, 4161, 7_200_000_000, 52)
        with patch.object(m, "_perf_read", return_value=perf):
            prompt_ns, eval_ns = m._split_durations(MagicMock(), 73_000_000_000, 4161, 52)

        assert prompt_ns == 60_100_000_000
        assert eval_ns == 7_200_000_000
        # Sanity: the resulting generation rate is physically possible.
        assert 52 / (eval_ns / 1e9) == pytest.approx(7.2, rel=0.01)

    def test_falls_back_to_estimate_without_counters(self):
        """Older backends report nothing; the approximation is better than
        zeroes, and it is the only reason the old code existed."""
        from hfl.engine import llama_cpp as m

        with patch.object(m, "_perf_read", return_value=None):
            prompt_ns, eval_ns = m._split_durations(MagicMock(), 100_000_000_000, 90, 10)

        assert prompt_ns == 90_000_000_000
        assert eval_ns == 10_000_000_000

    def test_zeroed_counters_fall_back_too(self):
        """A build that exposes the struct but never fills it must not make
        every request report 0 s of generation."""
        from hfl.engine import llama_cpp as m

        with patch.object(m, "_perf_read", return_value=(0, 0, 0, 0)):
            prompt_ns, eval_ns = m._split_durations(MagicMock(), 100_000_000_000, 50, 50)

        assert prompt_ns > 0 and eval_ns > 0

    def test_perf_helpers_never_raise(self):
        """They run on the hot generation path; a missing symbol must not
        turn a successful generation into a 500."""
        from hfl.engine import llama_cpp as m

        broken = MagicMock()
        del broken._ctx
        m._perf_reset(broken)  # must not raise
        assert m._perf_read(broken) is None


class TestContextReloadDecision:
    """A reload evicts 44 GiB and discards the KV cache, so it must only
    happen when the resident window genuinely cannot serve the request."""

    @pytest.fixture(autouse=True)
    def reset(self):
        reset_state()
        yield
        reset_state()

    @staticmethod
    def _state_with_resident(resident_ctx: int):
        engine = MagicMock()
        engine.context_size = resident_ctx
        manifest = MagicMock()
        manifest.name = "m"
        state = MagicMock()
        state.current_model = manifest
        state.engine = engine
        state.set_llm_engine = AsyncMock()
        state.ensure_llm_loaded = AsyncMock(return_value=(engine, manifest))
        return state, engine, manifest

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "resident,requested,reloads",
        [
            (32768, 8192, False),  # shrink  — the wasteful case, 6 of 16 observed
            (32768, 16384, False),  # shrink
            (16384, 8192, False),  # shrink
            (16384, 16384, False),  # exact match
            (8192, 16384, True),  # grow — genuinely needs a bigger window
            (16384, 32768, True),  # grow
        ],
    )
    @patch("hfl.api.model_loader.get_state")
    async def test_only_growing_the_window_reloads(
        self, mock_get_state, resident, requested, reloads
    ):
        state, engine, manifest = self._state_with_resident(resident)
        mock_get_state.return_value = state

        if not reloads:
            got_engine, got_manifest = await load_llm("m", num_ctx=requested)
            assert got_engine is engine and got_manifest is manifest
            state.ensure_llm_loaded.assert_not_called()
        else:
            with (
                patch("hfl.api.model_loader.get_registry") as reg,
                patch("hfl.api.model_loader.detect_model_type") as det,
                patch("hfl.api.model_loader.select_engine"),
            ):
                from hfl.converter.formats import ModelType

                reg.return_value.get.return_value = manifest
                det.return_value = ModelType.LLM
                manifest.local_path = "/mock/m.gguf"
                manifest.context_length = 0
                await load_llm("m", num_ctx=requested)
            state.ensure_llm_loaded.assert_called_once()

    @pytest.mark.asyncio
    @patch("hfl.api.model_loader.get_state")
    async def test_untracked_context_never_reloads(self, mock_get_state):
        """MLX/vLLM report 0 ("unknown"); reloading on that would thrash them
        on every request."""
        state, engine, manifest = self._state_with_resident(0)
        mock_get_state.return_value = state

        got_engine, _ = await load_llm("m", num_ctx=32768)
        assert got_engine is engine
        state.ensure_llm_loaded.assert_not_called()


class TestEnsureLoadedGuard:
    """``ensure_llm_loaded`` re-checks residency inside the per-model lock, so
    it needs the same rule or a concurrent request would undo the fix."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "resident,required,usable",
        [(32768, 8192, True), (32768, 32768, True), (8192, 32768, False), (0, 32768, True)],
    )
    async def test_resident_check_matches_load_llm(self, resident, required, usable):
        from hfl.api.state import ServerState

        state = ServerState()
        engine = MagicMock()
        engine.context_size = resident
        manifest = MagicMock()
        manifest.name = "m"
        state._engine = engine
        state._current_model = manifest

        loader = AsyncMock(return_value=(engine, manifest))
        with patch.object(state, "set_llm_engine", AsyncMock()):
            await state.ensure_llm_loaded("m", loader, required_ctx=required)

        assert loader.called is (not usable)
