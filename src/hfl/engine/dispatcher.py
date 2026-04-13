# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""In-server inference dispatcher with bounded queue (spec §5.3).

The llama.cpp and transformers-GPU backends share a single non-reentrant
model instance: two overlapping requests can corrupt each other's KV
cache and produce empty or truncated replies. HFL therefore serializes
inference via this :class:`InferenceDispatcher`, which exposes a
bounded-concurrency, bounded-wait-queue primitive:

- ``max_inflight``: how many requests may execute at once (default 1)
- ``max_queued``: how many more may wait in line (default 16)
- ``acquire_timeout``: cap on how long a caller waits for a slot before
  giving up with :class:`QueueTimeoutError` (default 60 s)

When the wait queue is full, callers are rejected immediately with
:class:`QueueFullError`, which the HTTP layer maps to 429 with a
``Retry-After`` header. Callers that time out waiting for a slot get
503 instead, signalling the server is simply saturated.

Metrics are exposed via :meth:`snapshot` so ``/healthz`` and the
per-response ``X-Queue-Depth`` header can report live state.

The dispatcher is async-aware. Synchronous (blocking) work must be run
on a thread via :func:`asyncio.to_thread` inside the supplied callable;
this module only manages the scheduling primitive, never the execution
context.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class QueueFullError(Exception):
    """Raised when the dispatcher's wait queue is full.

    Mapped to HTTP 429 by the route layer. The error is retryable — the
    caller should back off for ``retry_after_seconds`` and try again.
    """

    def __init__(self, depth: int, max_queued: int, retry_after: int):
        self.depth = depth
        self.max_queued = max_queued
        self.retry_after_seconds = retry_after
        super().__init__(
            f"inference queue full (depth={depth}, max={max_queued})"
        )


class QueueTimeoutError(Exception):
    """Raised when a caller waited too long for a dispatcher slot.

    Mapped to HTTP 503 by the route layer. Retryable.
    """

    def __init__(self, waited_seconds: float):
        self.waited_seconds = waited_seconds
        super().__init__(
            f"dispatcher slot acquire timed out after {waited_seconds:.1f}s"
        )


@dataclass(frozen=True)
class DispatcherSnapshot:
    """Immutable view of dispatcher state for metrics / health checks."""

    max_inflight: int
    max_queued: int
    in_flight: int
    depth: int
    accepted_total: int
    rejected_full_total: int
    rejected_timeout_total: int


class InferenceDispatcher:
    """Bounded-concurrency dispatcher for inference requests.

    See module docstring for behaviour. Implementation notes:

    - All mutations to ``in_flight`` / ``depth`` / counters happen under
      ``_counter_lock`` so fast-path and slow-path decisions observe a
      consistent view.
    - Concurrency enforcement is backed by an :class:`asyncio.Semaphore`
      whose capacity tracks ``max_inflight`` exactly.
    - Cancellation safety: if the caller is cancelled (or the surrounding
      ``with`` block raises) while waiting, the depth counter is
      unwound; if cancelled after acquiring a slot, the slot is released
      in the ``finally`` of :meth:`slot`.
    """

    def __init__(
        self,
        max_inflight: int = 1,
        max_queued: int = 16,
        acquire_timeout: float = 60.0,
    ) -> None:
        if max_inflight < 1:
            raise ValueError("max_inflight must be >= 1")
        if max_queued < 0:
            raise ValueError("max_queued must be >= 0")
        if acquire_timeout <= 0:
            raise ValueError("acquire_timeout must be > 0")

        self._max_inflight = max_inflight
        self._max_queued = max_queued
        self._acquire_timeout = acquire_timeout

        self._sem = asyncio.Semaphore(max_inflight)
        self._counter_lock = asyncio.Lock()

        self._in_flight = 0
        self._depth = 0
        self._accepted_total = 0
        self._rejected_full_total = 0
        self._rejected_timeout_total = 0

    # -- Introspection --------------------------------------------------

    @property
    def max_inflight(self) -> int:
        return self._max_inflight

    @property
    def max_queued(self) -> int:
        return self._max_queued

    @property
    def depth(self) -> int:
        """Number of callers currently waiting for a slot."""
        return self._depth

    @property
    def in_flight(self) -> int:
        """Number of callers currently executing."""
        return self._in_flight

    def snapshot(self) -> DispatcherSnapshot:
        return DispatcherSnapshot(
            max_inflight=self._max_inflight,
            max_queued=self._max_queued,
            in_flight=self._in_flight,
            depth=self._depth,
            accepted_total=self._accepted_total,
            rejected_full_total=self._rejected_full_total,
            rejected_timeout_total=self._rejected_timeout_total,
        )

    # -- Slot acquisition -----------------------------------------------

    @contextlib.asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Async context manager that holds a dispatch slot.

        Raises:
            QueueFullError: when the wait queue is already full. This
                decision is made *before* queueing so the client gets an
                immediate 429 instead of waiting.
            QueueTimeoutError: when the caller gave up waiting for a
                slot after ``acquire_timeout`` seconds.
        """
        # Phase 1 — decide fast vs slow path under the counter lock.
        async with self._counter_lock:
            if self._in_flight < self._max_inflight:
                # Fast path: there is capacity right now. Reserve the
                # slot in our bookkeeping; we will still acquire the
                # semaphore below to keep its value in sync.
                self._in_flight += 1
                fast_path = True
            else:
                # Slow path: must wait. Reject immediately if the wait
                # queue is already full.
                if self._depth >= self._max_queued:
                    self._rejected_full_total += 1
                    raise QueueFullError(
                        depth=self._depth,
                        max_queued=self._max_queued,
                        retry_after=max(1, int(self._acquire_timeout)),
                    )
                self._depth += 1
                fast_path = False

        # Phase 2 — acquire the semaphore. Outside the counter lock so
        # we do not deadlock other callers that are trying to release
        # their slot.
        start = time.monotonic()
        if fast_path:
            try:
                await self._sem.acquire()
            except BaseException:
                # Cancelled before the semaphore could be taken — roll
                # back the fast-path reservation.
                async with self._counter_lock:
                    self._in_flight -= 1
                raise
            async with self._counter_lock:
                self._accepted_total += 1
        else:
            try:
                await asyncio.wait_for(
                    self._sem.acquire(), timeout=self._acquire_timeout
                )
            except asyncio.TimeoutError:
                async with self._counter_lock:
                    self._depth -= 1
                    self._rejected_timeout_total += 1
                raise QueueTimeoutError(
                    waited_seconds=time.monotonic() - start
                )
            except BaseException:
                async with self._counter_lock:
                    self._depth -= 1
                raise
            async with self._counter_lock:
                self._depth -= 1
                self._in_flight += 1
                self._accepted_total += 1

        # Phase 3 — execute the guarded block.
        try:
            yield
        finally:
            # Phase 4 — release the slot. Release the semaphore BEFORE
            # decrementing ``in_flight`` would create a window where a
            # new fast-path caller observes capacity but sees the
            # semaphore still locked; doing it the other way round
            # creates the mirror window. Either is correct as long as
            # both happen; we pick decrement-then-release so snapshots
            # taken between the two observe conservative (higher)
            # in_flight, matching the spec's "don't under-count load".
            async with self._counter_lock:
                self._in_flight -= 1
            self._sem.release()

    async def run(
        self,
        func: Callable[[], Awaitable[T]],
    ) -> T:
        """Acquire a slot, execute ``func``, return its result.

        ``func`` must be a zero-argument async callable. Synchronous
        engine work should be wrapped with :func:`asyncio.to_thread`
        inside ``func``.
        """
        async with self.slot():
            return await func()

    # -- Administrative -------------------------------------------------

    def reset(self) -> None:
        """Reset counters and the internal semaphore.

        Only safe when there is nothing in flight (e.g. test teardown).
        Tests use this so each case starts clean without rebuilding the
        whole HFL container.
        """
        self._sem = asyncio.Semaphore(self._max_inflight)
        self._in_flight = 0
        self._depth = 0
        self._accepted_total = 0
        self._rejected_full_total = 0
        self._rejected_timeout_total = 0


def build_default_dispatcher() -> InferenceDispatcher:
    """Construct an :class:`InferenceDispatcher` from ``hfl.config``."""
    from hfl.config import config as hfl_config

    return InferenceDispatcher(
        max_inflight=hfl_config.queue_max_inflight,
        max_queued=hfl_config.queue_max_size,
        acquire_timeout=hfl_config.queue_acquire_timeout_seconds,
    )
