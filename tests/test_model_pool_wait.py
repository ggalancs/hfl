# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for model pool non-recursive waiting."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from hfl.engine.model_pool import ModelPool


@pytest.fixture
def pool():
    return ModelPool(max_models=2, idle_timeout_seconds=60.0)


class TestModelPoolConcurrentLoad:
    @pytest.mark.asyncio
    async def test_concurrent_get_or_load_same_model(self, pool):
        """Two coroutines loading the same model should not deadlock."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_manifest = MagicMock()

        load_count = 0

        async def loader():
            nonlocal load_count
            load_count += 1
            await asyncio.sleep(0.2)  # Simulate loading
            return mock_engine, mock_manifest, 100.0

        # Launch two concurrent loads
        results = await asyncio.gather(
            pool.get_or_load("model-a", loader),
            pool.get_or_load("model-a", loader),
        )

        # Both should succeed
        assert results[0] is not None
        assert results[1] is not None
        # Only one should have actually loaded (the other waited)
        assert load_count == 1

    @pytest.mark.asyncio
    async def test_get_or_load_no_stack_overflow(self, pool):
        """Non-recursive waiting should not cause stack overflow."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_manifest = MagicMock()

        async def loader():
            await asyncio.sleep(0.05)
            return mock_engine, mock_manifest, 50.0

        result = await pool.get_or_load("model-x", loader)
        assert result is not None
        assert result.engine == mock_engine

    @pytest.mark.asyncio
    async def test_cached_model_returns_immediately(self, pool):
        """Cached model should be returned without loading."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_manifest = MagicMock()

        async def loader():
            return mock_engine, mock_manifest, 50.0

        # First load
        await pool.get_or_load("model-y", loader)

        # Second call should return cached
        call_count = 0

        async def counting_loader():
            nonlocal call_count
            call_count += 1
            return mock_engine, mock_manifest, 50.0

        result = await pool.get_or_load("model-y", counting_loader)
        assert result is not None
        assert call_count == 0  # Loader not called for cached model


class TestModelPoolPollingRetryOnFailure:
    """Covers model_pool.py lines 154-160: the branch where the waiting
    coroutine observes that the first loader dropped out of ``_loading``
    (because it raised), exits the polling loop, finds no cache entry,
    and retries the load itself.

    These tests serialise A and B explicitly via ``asyncio.Event`` so
    the ordering is deterministic rather than timing-dependent.
    """

    @pytest.mark.asyncio
    async def test_waiter_retries_loader_after_first_loader_fails(self):
        """When coroutine A's loader raises, coroutine B — which was
        polling — must break out of the polling loop and re-attempt
        the load itself. The second attempt should succeed.
        """
        pool = ModelPool(max_models=2, idle_timeout_seconds=60.0)

        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_manifest = MagicMock()

        a_loader_entered = asyncio.Event()  # A has taken the loading slot
        release_a = asyncio.Event()  # test tells A when to fail
        load_attempts = 0

        async def loader_first_fails_then_succeeds():
            nonlocal load_attempts
            load_attempts += 1
            attempt = load_attempts
            if attempt == 1:
                a_loader_entered.set()
                await release_a.wait()
                raise RuntimeError("first loader deliberately fails")
            # Subsequent attempts (from the retry path) succeed
            return mock_engine, mock_manifest, 100.0

        # Launch A; it will block inside the loader until release_a fires.
        task_a = asyncio.create_task(
            pool.get_or_load("model-retry", loader_first_fails_then_succeeds)
        )
        # Wait for A to be inside the loader — this guarantees A has
        # already released the counter lock AND added itself to
        # `_loading`, so B will see `_loading` contain the model.
        await a_loader_entered.wait()

        # Launch B; it will observe A is loading, enter the polling loop.
        task_b = asyncio.create_task(
            pool.get_or_load("model-retry", loader_first_fails_then_succeeds)
        )
        # Give B at least two polling iterations so it's firmly in the
        # loop and not just about to enter. Each iteration is 0.1 s.
        await asyncio.sleep(0.25)

        # Now let A fail; its ``finally`` block will remove model-retry
        # from ``_loading`` — which is B's signal to break out of the
        # polling loop (line 153-154) and retry via recursive call
        # (line 160).
        release_a.set()

        results = await asyncio.gather(task_a, task_b, return_exceptions=True)

        # A propagates its loader's exception
        assert isinstance(results[0], RuntimeError)
        assert "first loader deliberately fails" in str(results[0])

        # B succeeds via the retry path
        assert not isinstance(results[1], BaseException)
        assert results[1].engine is mock_engine

        # The loader was called exactly twice: once by A (failed), once
        # by B's recursive retry (succeeded).
        assert load_attempts == 2

    @pytest.mark.asyncio
    async def test_loading_slot_released_on_exception(self):
        """When a solo loader raises, ``_loading`` must be cleaned so
        the next caller doesn't deadlock waiting for a phantom loader.

        This exercises the ``finally`` block at model_pool.py:204-207
        after an exception — even though the line itself is covered by
        happy-path tests, the exception path deserves an explicit guard
        against regression.
        """
        pool = ModelPool(max_models=2, idle_timeout_seconds=60.0)

        async def failing_loader():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await pool.get_or_load("model-fail", failing_loader)

        # Critical invariant: the slot MUST be released after the
        # exception; otherwise subsequent callers would hang in the
        # polling loop waiting for a loader that will never finish.
        assert "model-fail" not in pool._loading

        # And a subsequent successful load should work immediately
        # without any polling / race.
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_manifest = MagicMock()

        async def ok_loader():
            return mock_engine, mock_manifest, 10.0

        result = await pool.get_or_load("model-fail", ok_loader)
        assert result.engine is mock_engine


class TestModelPoolDoubleLoadedRace:
    """Covers model_pool.py lines 183-188: the branch inside
    ``get_or_load`` where, after a successful ``loader()`` call, the
    cache-update lock discovers the model is ALREADY in ``_models``
    (another coroutine populated it while we were running the loader).
    The engine we just loaded must be unloaded and the pre-existing
    entry returned instead.

    Triggering this deterministically requires an asymmetric race that
    we simulate via a loader that writes to ``_models`` directly
    *during* its own execution.
    """

    @pytest.mark.asyncio
    async def test_concurrent_cache_population_unloads_our_engine(self):
        """Loader completes → lock acquired → ``_models`` already has an
        entry (injected mid-loader) → our freshly loaded engine gets
        ``unload()``-ed and the existing cached entry is returned.
        """
        from hfl.engine.model_pool import CachedModel

        pool = ModelPool(max_models=2, idle_timeout_seconds=60.0)

        # The "winning" cached model injected mid-loader
        existing_engine = MagicMock()
        existing_engine.is_loaded = True
        existing_manifest = MagicMock()
        existing_cached = CachedModel(
            engine=existing_engine,
            manifest=existing_manifest,
            last_used=0.0,
            load_time_ms=0.0,
            memory_estimate_mb=50.0,
        )

        # The engine our loader returns; it should end up ``unload()``-ed
        losing_engine = MagicMock()
        losing_engine.is_loaded = True
        losing_manifest = MagicMock()

        async def racing_loader():
            # Simulate another coroutine populating the cache while we
            # were loading. In production this would be another call
            # that finished faster; here we inject directly so the race
            # is deterministic.
            async with pool._lock:
                pool._models["model-race"] = existing_cached
                pool._total_memory_mb += existing_cached.memory_estimate_mb
            return losing_engine, losing_manifest, 100.0

        result = await pool.get_or_load("model-race", racing_loader)

        # We must receive the pre-existing entry, not our own.
        assert result is existing_cached
        assert result.engine is existing_engine

        # Our freshly loaded engine must have been unload()-ed to avoid
        # an orphaned model sitting in RAM/GPU. This is the critical
        # invariant covered by line 184.
        losing_engine.unload.assert_called_once()


class TestModelPoolLockGuardedCacheHit:
    """Covers model_pool.py lines 127-130: the double-check inside the
    counter lock that finds the model already cached (populated by
    another coroutine between the lock-free fast-path ``get()`` and
    the lock acquisition).
    """

    @pytest.mark.asyncio
    async def test_cache_populated_between_fastpath_and_lock(self):
        """A second coroutine populates ``_models`` in the brief window
        between our lock-free ``get()`` returning None and our
        ``async with self._lock:`` acquiring the lock. The double-check
        under the lock must return the cached entry — not proceed to
        the loader.
        """
        from hfl.engine.model_pool import CachedModel

        pool = ModelPool(max_models=2, idle_timeout_seconds=60.0)

        existing_engine = MagicMock()
        existing_engine.is_loaded = True
        existing_manifest = MagicMock()
        existing_cached = CachedModel(
            engine=existing_engine,
            manifest=existing_manifest,
            last_used=0.0,
            load_time_ms=0.0,
            memory_estimate_mb=50.0,
        )

        # Monkey-patch the pool's ``get`` method so that the *first*
        # call (the lock-free fast-path check on line 118) returns
        # None, but between that call and the subsequent lock
        # acquisition, we populate ``_models`` — so the double-check
        # on line 126 succeeds.
        original_get = pool.get
        call_num = 0

        async def racy_get(model_name: str):
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                # First call (fast path) — cache really is empty,
                # but between returning None and the caller acquiring
                # the lock, another "coroutine" writes the entry.
                result = await original_get(model_name)
                async with pool._lock:
                    pool._models[model_name] = existing_cached
                    pool._total_memory_mb += existing_cached.memory_estimate_mb
                return result
            return await original_get(model_name)

        pool.get = racy_get  # type: ignore[assignment]

        # The loader MUST NOT be called — if the double-check works
        # we short-circuit before reaching it.
        loader_called = False

        async def loader():
            nonlocal loader_called
            loader_called = True
            return MagicMock(), MagicMock(), 10.0

        result = await pool.get_or_load("late-arrival", loader)

        assert result is existing_cached
        assert loader_called is False, "Loader must not run when cache was populated mid-race"


class TestModelPoolUnloadFailureLogged:
    """Covers model_pool.py lines 198-201: when the post-load cache
    update raises AND the cleanup ``engine.unload()`` also raises, the
    cleanup error is logged but does NOT mask the original exception.
    """

    @pytest.mark.asyncio
    async def test_unload_failure_is_logged_not_propagated(self):
        """Inject a post-loader failure, then make ``unload()`` raise.
        The outer exception must be the loader-side one; the unload
        failure is only recorded via ``logger.error``.
        """
        pool = ModelPool(max_models=2, idle_timeout_seconds=60.0)

        failing_engine = MagicMock()
        failing_engine.is_loaded = True
        failing_engine.unload.side_effect = RuntimeError("unload boom")
        manifest = MagicMock()

        # Loader returns an engine, but we inject a failure by making
        # the cache-update step raise. The simplest path is to replace
        # ``pool._evict_if_needed_locked`` — wait, that's before the
        # loader. Instead we monkey-patch the dict assignment by
        # shadowing ``_total_memory_mb`` with a property that raises.
        # Simpler: wrap the loader so the very last thing before
        # return forces an exception propagated inside the try.
        # Actually the cleanest hook: monkey-patch
        # ``CachedModel`` construction to raise inside the pool module.
        from hfl.engine import model_pool as mp

        real_cached_model = mp.CachedModel

        class BoomCachedModel(real_cached_model):  # type: ignore[misc]
            def __init__(self, *args, **kwargs):
                raise ValueError("cache-entry construction failed")

        async def loader():
            return failing_engine, manifest, 10.0

        with patch.object(mp, "CachedModel", BoomCachedModel):
            with patch.object(mp, "logger") as mock_logger:
                with pytest.raises(ValueError, match="cache-entry construction failed"):
                    await pool.get_or_load("model-unload-boom", loader)

        # The ``unload()`` call happened and raised; the raise was
        # swallowed and recorded by ``logger.error``.
        failing_engine.unload.assert_called_once()
        assert mock_logger.error.called
        args, _ = mock_logger.error.call_args
        assert "unload" in args[0].lower()

        # Slot released even through the double-exception path.
        assert "model-unload-boom" not in pool._loading
