# SPDX-License-Identifier: HRUL-1.0
"""Unit tests for :mod:`hfl.engine.dispatcher` (spec §5.3).

Covers:

- basic acceptance / serialization guarantees
- rejection when the wait queue is full (``QueueFullError`` → 429)
- rejection when a slot acquire times out (``QueueTimeoutError`` → 503)
- metrics: ``in_flight``, ``depth``, accepted/rejected counters
- cancellation / exception safety: slots are always released
- streaming use case: ``async with slot()`` holds capacity for the
  whole lifetime of the block
- construction / validation edge cases
"""

from __future__ import annotations

import asyncio

import pytest

from hfl.engine.dispatcher import (
    DispatcherSnapshot,
    InferenceDispatcher,
    QueueFullError,
    QueueTimeoutError,
    build_default_dispatcher,
)


# --- Construction -------------------------------------------------------------


class TestConstruction:
    def test_defaults_match_spec(self):
        d = InferenceDispatcher()
        assert d.max_inflight == 1
        assert d.max_queued == 16
        assert d.in_flight == 0
        assert d.depth == 0

    def test_custom_values(self):
        d = InferenceDispatcher(max_inflight=4, max_queued=32, acquire_timeout=5)
        assert d.max_inflight == 4
        assert d.max_queued == 32

    @pytest.mark.parametrize("bad", [-1, 0])
    def test_rejects_bad_max_inflight(self, bad):
        with pytest.raises(ValueError):
            InferenceDispatcher(max_inflight=bad)

    def test_rejects_bad_max_queued(self):
        with pytest.raises(ValueError):
            InferenceDispatcher(max_queued=-1)

    def test_rejects_bad_timeout(self):
        with pytest.raises(ValueError):
            InferenceDispatcher(acquire_timeout=0)

    def test_build_default_uses_config(self):
        d = build_default_dispatcher()
        assert d.max_inflight >= 1
        assert d.max_queued >= 0


# --- Basic run ----------------------------------------------------------------


class TestRun:
    async def test_run_returns_result(self):
        d = InferenceDispatcher(max_inflight=1, max_queued=4)

        async def _work():
            return 42

        assert await d.run(_work) == 42

    async def test_run_propagates_exception(self):
        d = InferenceDispatcher(max_inflight=1, max_queued=4)

        async def _boom():
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError, match="nope"):
            await d.run(_boom)

    async def test_run_releases_slot_on_exception(self):
        """Regression: a crashing handler must not leave the slot held."""
        d = InferenceDispatcher(max_inflight=1, max_queued=4)

        async def _boom():
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            await d.run(_boom)

        # Slot must be free again
        assert d.in_flight == 0
        assert d.depth == 0

        async def _ok():
            return "ok"

        assert await d.run(_ok) == "ok"

    async def test_counters_after_successful_run(self):
        d = InferenceDispatcher(max_inflight=1, max_queued=4)

        async def _work():
            return 1

        for _ in range(5):
            await d.run(_work)

        snap = d.snapshot()
        assert snap.accepted_total == 5
        assert snap.rejected_full_total == 0
        assert snap.rejected_timeout_total == 0
        assert snap.in_flight == 0
        assert snap.depth == 0


# --- Serialization ------------------------------------------------------------


class TestSerialization:
    async def test_concurrent_calls_execute_one_at_a_time(self):
        """With max_inflight=1 the dispatcher must observe strict
        serialization: at no point are two bodies running at once."""
        d = InferenceDispatcher(max_inflight=1, max_queued=8)
        observed_max_inflight = 0
        concurrent = 0
        lock = asyncio.Lock()

        async def _work():
            nonlocal concurrent, observed_max_inflight
            async with lock:
                concurrent += 1
                observed_max_inflight = max(observed_max_inflight, concurrent)
            await asyncio.sleep(0.02)
            async with lock:
                concurrent -= 1

        await asyncio.gather(*(d.run(_work) for _ in range(5)))
        assert observed_max_inflight == 1

    async def test_max_inflight_two_allows_exactly_two_in_parallel(self):
        d = InferenceDispatcher(max_inflight=2, max_queued=8)
        observed_max_inflight = 0
        concurrent = 0
        lock = asyncio.Lock()

        async def _work():
            nonlocal concurrent, observed_max_inflight
            async with lock:
                concurrent += 1
                observed_max_inflight = max(observed_max_inflight, concurrent)
            await asyncio.sleep(0.05)
            async with lock:
                concurrent -= 1

        await asyncio.gather(*(d.run(_work) for _ in range(6)))
        assert observed_max_inflight == 2

    async def test_fifo_order_of_completion_under_serialization(self):
        d = InferenceDispatcher(max_inflight=1, max_queued=8)
        order: list[int] = []

        async def _work(i):
            await asyncio.sleep(0.01)
            order.append(i)

        async def _run(i):
            await d.run(lambda i=i: _work(i))

        await asyncio.gather(*(_run(i) for i in range(5)))
        # Not strictly FIFO in asyncio.Semaphore, but every value must
        # appear exactly once.
        assert sorted(order) == [0, 1, 2, 3, 4]


# --- Queue full (rejected immediately) ---------------------------------------


class TestQueueFull:
    async def test_queue_full_raises_queue_full_error(self):
        d = InferenceDispatcher(max_inflight=1, max_queued=1, acquire_timeout=5)
        gate = asyncio.Event()

        async def _slow():
            await gate.wait()

        # Fill one in-flight + one waiter.
        t1 = asyncio.create_task(d.run(_slow))
        await asyncio.sleep(0.01)

        async def _waiter():
            async with d.slot():
                pass

        t2 = asyncio.create_task(_waiter())
        # Give t2 time to enter the queue.
        await asyncio.sleep(0.01)
        assert d.depth == 1

        # Third caller must be rejected.
        with pytest.raises(QueueFullError) as exc:
            async with d.slot():
                pass
        assert exc.value.depth == 1
        assert exc.value.max_queued == 1
        assert exc.value.retry_after_seconds > 0

        # Release the in-flight one so both tasks can finish.
        gate.set()
        await asyncio.gather(t1, t2)

        snap = d.snapshot()
        assert snap.rejected_full_total == 1
        assert snap.accepted_total == 2

    async def test_queue_full_does_not_leak_depth(self):
        d = InferenceDispatcher(max_inflight=1, max_queued=0, acquire_timeout=5)
        gate = asyncio.Event()

        async def _slow():
            await gate.wait()

        t1 = asyncio.create_task(d.run(_slow))
        await asyncio.sleep(0.01)

        # With max_queued=0 and in_flight==1, the next caller is
        # rejected without even entering the depth counter.
        for _ in range(3):
            with pytest.raises(QueueFullError):
                async with d.slot():
                    pass

        assert d.depth == 0
        gate.set()
        await t1


# --- Acquire timeout (rejected after waiting) --------------------------------


class TestAcquireTimeout:
    async def test_slot_acquire_times_out(self):
        d = InferenceDispatcher(
            max_inflight=1, max_queued=4, acquire_timeout=0.1
        )
        gate = asyncio.Event()

        async def _slow():
            await gate.wait()

        t1 = asyncio.create_task(d.run(_slow))
        await asyncio.sleep(0.01)

        with pytest.raises(QueueTimeoutError) as exc:
            async with d.slot():
                pass
        assert exc.value.waited_seconds >= 0.05

        snap = d.snapshot()
        assert snap.rejected_timeout_total == 1
        # Depth must be unwound after the rejection.
        assert snap.depth == 0

        gate.set()
        await t1


# --- Cancellation safety ------------------------------------------------------


class TestCancellation:
    async def test_cancel_while_waiting_releases_depth(self):
        d = InferenceDispatcher(
            max_inflight=1, max_queued=4, acquire_timeout=5
        )
        gate = asyncio.Event()

        async def _slow():
            await gate.wait()

        t1 = asyncio.create_task(d.run(_slow))
        await asyncio.sleep(0.01)

        async def _cancel_me():
            async with d.slot():
                pass  # pragma: no cover

        t2 = asyncio.create_task(_cancel_me())
        # Let t2 enter the wait queue.
        await asyncio.sleep(0.01)
        assert d.depth == 1
        t2.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t2
        assert d.depth == 0

        gate.set()
        await t1

    async def test_cancel_holding_slot_releases_slot(self):
        d = InferenceDispatcher(max_inflight=1, max_queued=4)

        slot_entered = asyncio.Event()
        release_me = asyncio.Event()

        async def _long():
            async with d.slot():
                slot_entered.set()
                await release_me.wait()

        t = asyncio.create_task(_long())
        await slot_entered.wait()
        assert d.in_flight == 1
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        assert d.in_flight == 0


# --- Snapshot & metrics --------------------------------------------------------


class TestSnapshot:
    async def test_snapshot_live_during_call(self):
        d = InferenceDispatcher(max_inflight=1, max_queued=4)
        inside = asyncio.Event()
        finish = asyncio.Event()

        async def _work():
            inside.set()
            await finish.wait()

        t = asyncio.create_task(d.run(_work))
        await inside.wait()

        snap = d.snapshot()
        assert snap.in_flight == 1
        assert snap.depth == 0
        assert snap.max_inflight == 1

        finish.set()
        await t
        assert d.snapshot().in_flight == 0

    async def test_snapshot_is_dataclass_frozen(self):
        d = InferenceDispatcher()
        snap = d.snapshot()
        assert isinstance(snap, DispatcherSnapshot)
        with pytest.raises(Exception):
            snap.in_flight = 99  # type: ignore[misc]


# --- Reset --------------------------------------------------------------------


class TestReset:
    async def test_reset_clears_counters(self):
        d = InferenceDispatcher(max_inflight=1, max_queued=1)

        async def _work():
            return None

        await d.run(_work)
        assert d.snapshot().accepted_total == 1
        d.reset()
        snap = d.snapshot()
        assert snap.accepted_total == 0
        assert snap.in_flight == 0
        assert snap.depth == 0


# --- Streaming-shaped usage ---------------------------------------------------


class TestStreamingShape:
    async def test_slot_held_across_iteration(self):
        """A streaming caller uses ``async with slot()`` to hold the
        slot while it iterates a generator. While it holds the slot,
        no other caller may run (with max_inflight=1)."""
        d = InferenceDispatcher(max_inflight=1, max_queued=4)
        stream_started = asyncio.Event()
        stream_midpoint = asyncio.Event()
        other_done = asyncio.Event()

        async def _stream():
            async with d.slot():
                stream_started.set()
                # Pretend to yield tokens for a while.
                await asyncio.sleep(0)
                await stream_midpoint.wait()
                assert not other_done.is_set(), (
                    "another caller ran while we held the slot"
                )

        async def _other():
            async with d.slot():
                other_done.set()

        t_stream = asyncio.create_task(_stream())
        await stream_started.wait()
        t_other = asyncio.create_task(_other())
        await asyncio.sleep(0.02)  # give _other a chance if it could run
        assert not other_done.is_set()
        stream_midpoint.set()
        await asyncio.gather(t_stream, t_other)
        assert other_done.is_set()
