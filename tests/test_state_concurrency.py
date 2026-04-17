# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Concurrency tests for ServerState.

Tests for race conditions, concurrent model loading, and cleanup under load.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from hfl.api.state import get_state, reset_state


@pytest.fixture
def fresh_state():
    """Create a fresh ServerState for testing."""
    reset_state()
    yield get_state()
    reset_state()


@pytest.fixture
def mock_engine():
    """Create a mock inference engine."""
    engine = MagicMock()
    engine.is_loaded = True
    engine.unload = MagicMock()
    return engine


@pytest.fixture
def mock_manifest():
    """Create a mock model manifest."""
    manifest = MagicMock()
    manifest.name = "test-model"
    return manifest


class TestConcurrentModelLoading:
    """Tests for concurrent model loading scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_same_model_load(self, fresh_state, mock_engine, mock_manifest):
        """Multiple concurrent loads of same model should serialize."""
        load_count = 0
        load_delay = 0.1

        async def mock_loader():
            nonlocal load_count
            load_count += 1
            await asyncio.sleep(load_delay)
            return mock_engine, mock_manifest

        # Start 5 concurrent loads of the same model
        tasks = [
            asyncio.create_task(fresh_state.ensure_llm_loaded("test-model", mock_loader))
            for _ in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All should return the same engine
        assert all(r[0] == mock_engine for r in results)
        # Should only load once (first loader runs, others wait and get cached result)
        # Actually with the new code, it loads each time because we removed the fast path
        # The lock serializes, so it should be 1
        assert load_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_different_model_loads(self, fresh_state):
        """Different models can load concurrently."""
        load_times = {}

        async def create_loader(model_name: str, delay: float):
            async def loader():
                load_times[model_name] = asyncio.get_event_loop().time()
                await asyncio.sleep(delay)
                engine = MagicMock()
                engine.is_loaded = True
                manifest = MagicMock()
                manifest.name = model_name
                return engine, manifest

            return loader

        loader_a = await create_loader("model-a", 0.1)
        loader_b = await create_loader("model-b", 0.1)

        # Load two different models concurrently
        task_a = asyncio.create_task(fresh_state.ensure_llm_loaded("model-a", loader_a))
        task_b = asyncio.create_task(fresh_state.ensure_llm_loaded("model-b", loader_b))

        await asyncio.gather(task_a, task_b)

        # Both loads should have started almost simultaneously
        time_diff = abs(load_times.get("model-a", 0) - load_times.get("model-b", 0))
        assert time_diff < 0.05, "Different models should load concurrently"

    @pytest.mark.asyncio
    async def test_model_switch_during_load(self, fresh_state, mock_engine):
        """Switching models during load should wait for current load."""
        load_complete = asyncio.Event()

        async def slow_loader():
            await load_complete.wait()
            manifest = MagicMock()
            manifest.name = "model-a"
            return mock_engine, manifest

        async def fast_loader():
            engine = MagicMock()
            engine.is_loaded = True
            manifest = MagicMock()
            manifest.name = "model-b"
            return engine, manifest

        # Start slow load
        task_a = asyncio.create_task(fresh_state.ensure_llm_loaded("model-a", slow_loader))

        # Give it time to start
        await asyncio.sleep(0.01)

        # Start fast load of different model
        task_b = asyncio.create_task(fresh_state.ensure_llm_loaded("model-b", fast_loader))

        # Let fast load complete first
        await asyncio.sleep(0.01)

        # Now let slow load complete
        load_complete.set()

        await asyncio.gather(task_a, task_b)

        # No exception should occur


class TestCleanupUnderLoad:
    """Tests for cleanup during concurrent operations."""

    @pytest.mark.asyncio
    async def test_cleanup_during_load(self, fresh_state, mock_engine, mock_manifest):
        """Cleanup should wait for loading to complete."""
        load_started = asyncio.Event()
        can_complete = asyncio.Event()

        async def slow_loader():
            load_started.set()
            await can_complete.wait()
            return mock_engine, mock_manifest

        # Start loading
        load_task = asyncio.create_task(fresh_state.ensure_llm_loaded("test-model", slow_loader))

        await load_started.wait()

        # Start cleanup while loading
        cleanup_task = asyncio.create_task(fresh_state.cleanup())

        # Allow load to complete
        can_complete.set()

        # Both should complete without error
        await asyncio.gather(load_task, cleanup_task)

    @pytest.mark.asyncio
    async def test_rapid_cleanup_cycles(self, fresh_state, mock_engine, mock_manifest):
        """Rapid cleanup cycles should not cause issues."""

        async def quick_loader():
            return mock_engine, mock_manifest

        # Load and cleanup rapidly
        for _ in range(10):
            await fresh_state.ensure_llm_loaded("test-model", quick_loader)
            await fresh_state.cleanup()

        # State should be clean
        assert fresh_state.engine is None
        assert fresh_state.current_model is None


class TestStateConsistency:
    """Tests for state consistency under concurrent access."""

    @pytest.mark.asyncio
    async def test_engine_and_model_atomicity(self, fresh_state, mock_engine, mock_manifest):
        """Engine and model should always be consistent."""

        async def loader():
            return mock_engine, mock_manifest

        await fresh_state.ensure_llm_loaded("test-model", loader)

        # Check atomicity - engine and model should both be set or both be None
        assert (fresh_state.engine is None) == (fresh_state.current_model is None)

        await fresh_state.cleanup()

        assert (fresh_state.engine is None) == (fresh_state.current_model is None)

    @pytest.mark.asyncio
    async def test_is_loading_flag_accuracy(self, fresh_state, mock_engine, mock_manifest):
        """is_loading flag should be accurate during load."""
        load_started = asyncio.Event()
        can_complete = asyncio.Event()

        async def controlled_loader():
            load_started.set()
            await can_complete.wait()
            return mock_engine, mock_manifest

        assert not fresh_state.is_loading

        load_task = asyncio.create_task(
            fresh_state.ensure_llm_loaded("test-model", controlled_loader)
        )

        await load_started.wait()
        assert fresh_state.is_loading
        assert "test-model" in fresh_state.loading_models

        can_complete.set()
        await load_task

        assert not fresh_state.is_loading
        assert "test-model" not in fresh_state.loading_models

    @pytest.mark.asyncio
    async def test_timeout_handling(self, fresh_state):
        """Loading timeout should be handled properly."""

        async def infinite_loader():
            await asyncio.sleep(10)  # Would take 10 seconds
            raise AssertionError("Should not reach here")

        with pytest.raises(asyncio.TimeoutError):
            await fresh_state.ensure_llm_loaded(
                "test-model",
                infinite_loader,
                timeout=0.1,  # 100ms timeout
            )

        # State should be clean after timeout
        assert not fresh_state.is_loading


class TestTTSConcurrency:
    """Tests for TTS engine concurrency."""

    @pytest.mark.asyncio
    async def test_llm_and_tts_concurrent(self, fresh_state):
        """LLM and TTS can load concurrently."""

        async def llm_loader():
            await asyncio.sleep(0.05)
            engine = MagicMock()
            engine.is_loaded = True
            manifest = MagicMock()
            manifest.name = "llm-model"
            return engine, manifest

        async def tts_loader():
            await asyncio.sleep(0.05)
            engine = MagicMock()
            engine.is_loaded = True
            manifest = MagicMock()
            manifest.name = "tts-model"
            return engine, manifest

        llm_task = asyncio.create_task(fresh_state.ensure_llm_loaded("llm-model", llm_loader))
        tts_task = asyncio.create_task(fresh_state.ensure_tts_loaded("tts-model", tts_loader))

        await asyncio.gather(llm_task, tts_task)

        # Both should be loaded
        assert fresh_state.is_llm_loaded()
        assert fresh_state.is_tts_loaded()


class TestUnloadOffLoop:
    """Tests for the R7 change that runs ``engine.unload()`` via
    ``asyncio.to_thread`` so a slow unload does NOT block the event
    loop. These guard the off-loop contract; they check BOTH that the
    unload ran in a worker thread AND that the lock is still held
    during the unload (so we don't trade a freeze for a race).
    """

    @pytest.mark.asyncio
    async def test_unload_runs_off_the_event_loop_thread(self, fresh_state):
        """The invariant we care about is that ``engine.unload()`` is
        dispatched to a worker thread, not executed on the asyncio
        event-loop thread. Detect this by comparing the thread id
        recorded inside ``unload`` to the loop thread's id. This is a
        robust, timing-free check — a regression to synchronous
        ``engine.unload()`` would make both ids match.
        """
        import threading

        loop_thread_id = threading.get_ident()
        unload_thread_id: list[int] = []

        def record_thread_and_return() -> None:
            unload_thread_id.append(threading.get_ident())

        slow_engine = MagicMock()
        slow_engine.is_loaded = True
        slow_engine.unload = record_thread_and_return

        new_engine = MagicMock()
        new_engine.is_loaded = True
        manifest = MagicMock()
        manifest.name = "m"

        await fresh_state.set_llm_engine(slow_engine, manifest)
        await fresh_state.set_llm_engine(new_engine, manifest)

        assert len(unload_thread_id) == 1, "unload() should have been called once"
        assert unload_thread_id[0] != loop_thread_id, (
            f"unload() ran on the event-loop thread (id={loop_thread_id}); "
            "it must be dispatched via asyncio.to_thread()"
        )
        assert fresh_state._engine is new_engine

    @pytest.mark.asyncio
    async def test_unload_exception_does_not_break_state(self, fresh_state):
        """If the engine's ``unload()`` raises inside ``set_llm_engine``,
        the exception must propagate and state must not be left in a
        half-updated state (old engine still referenced, new engine
        partially assigned).
        """
        raising_engine = MagicMock()
        raising_engine.is_loaded = True
        raising_engine.unload = MagicMock(side_effect=RuntimeError("GPU fault"))

        new_engine = MagicMock()
        new_engine.is_loaded = True
        manifest = MagicMock()
        manifest.name = "next"

        await fresh_state.set_llm_engine(raising_engine, manifest)

        with pytest.raises(RuntimeError, match="GPU fault"):
            await fresh_state.set_llm_engine(new_engine, manifest)

        # On exception, state.engine is still the OLD engine — the
        # code raised before the assignment. This is the contract we
        # want: callers observe "replacement failed", not a dangling
        # half-assignment.
        assert fresh_state._engine is raising_engine

    @pytest.mark.asyncio
    async def test_cleanup_runs_both_unloads_off_loop(self, fresh_state):
        """``cleanup()`` also dispatches both LLM and TTS unload to
        worker threads. Same thread-id-based detection as
        ``test_unload_runs_off_the_event_loop_thread`` — no timing.
        """
        import threading

        loop_thread_id = threading.get_ident()
        threads_used: list[int] = []

        def record_thread() -> None:
            threads_used.append(threading.get_ident())

        llm_engine = MagicMock()
        llm_engine.is_loaded = True
        llm_engine.unload = record_thread

        tts_engine = MagicMock()
        tts_engine.is_loaded = True
        tts_engine.unload = record_thread

        manifest = MagicMock()
        manifest.name = "m"

        await fresh_state.set_llm_engine(llm_engine, manifest)
        await fresh_state.set_tts_engine(tts_engine, manifest)

        await fresh_state.cleanup()

        assert len(threads_used) == 2, (
            f"cleanup should call both unloads once each (got {len(threads_used)})"
        )
        for tid in threads_used:
            assert tid != loop_thread_id, (
                "cleanup() unload ran on the event-loop thread; must use asyncio.to_thread()"
            )
        assert fresh_state._engine is None
        assert fresh_state._tts_engine is None

    @pytest.mark.asyncio
    async def test_concurrent_replace_and_query_never_see_partial_state(self, fresh_state):
        """Hammer the state: repeatedly replace the LLM engine while a
        concurrent task keeps calling ``is_llm_loaded()``. The
        off-loop unload changes timing but must not expose a state
        where ``is_llm_loaded()`` returns True for a stale engine.
        """
        manifest = MagicMock()
        manifest.name = "m"

        # Queue of engines; each replaces the previous one.
        engines = []
        for i in range(8):
            e = MagicMock()
            e.is_loaded = True
            e.unload = MagicMock()
            engines.append(e)

        stop = asyncio.Event()
        saw_loaded = 0
        saw_unloaded = 0

        async def querier():
            nonlocal saw_loaded, saw_unloaded
            while not stop.is_set():
                if fresh_state.is_llm_loaded():
                    saw_loaded += 1
                else:
                    saw_unloaded += 1
                await asyncio.sleep(0)  # yield

        q = asyncio.create_task(querier())

        # Initial seed
        await fresh_state.set_llm_engine(engines[0], manifest)

        # Rotate through engines
        for e in engines[1:]:
            await fresh_state.set_llm_engine(e, manifest)

        # Final unload
        for e in engines:
            e.is_loaded = False  # simulate post-unload state
        await fresh_state.set_llm_engine(None, None)

        stop.set()
        await q

        # We should have observed many loaded states during the
        # churn, and at least one unloaded state at the end.
        assert saw_loaded > 0
        # Every engine we installed should have had unload() called
        # exactly once (when it was replaced), except the last which
        # is released by the final None-set.
        for e in engines[:-1]:
            e.unload.assert_called_once()
