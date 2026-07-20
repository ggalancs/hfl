# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Streaming utilities with backpressure and timeout support.

This module provides helpers for streaming responses that:
- Apply backpressure to prevent memory exhaustion
- Support timeout for hung streams
- Detect client disconnections (when Request is available)
- Emit heartbeats for long-running streams
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterator

from hfl.config import config as _hfl_config

if TYPE_CHECKING:
    from starlette.requests import Request

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_STREAM_TIMEOUT = 300.0  # 5 minutes
DEFAULT_QUEUE_SIZE = 100  # Max tokens in buffer
DEFAULT_HEARTBEAT_INTERVAL = 30.0  # Seconds between heartbeats


class StreamTimeoutError(Exception):
    """Raised when a stream operation times out."""


class StreamCancelledError(Exception):
    """Raised when a stream is cancelled (e.g., client disconnect)."""


def _close_iterator(it: Iterator[Any]) -> None:
    """Close a generator-backed iterator if it supports ``close()``.

    Called from the producer thread (which drives ``next()``), so it is safe:
    closing propagates ``GeneratorExit`` into the engine's ``generate_stream``,
    letting it run its cooperative cancel / cleanup instead of leaking the
    worker thread until GC (CON-3). A no-op for plain iterators.
    """
    close = getattr(it, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # pragma: no cover - defensive
            pass


def _record_stream_orphan() -> None:
    """Bump the SSE/HTTP orphan counter (parity with the WS path)."""
    try:
        from hfl.metrics import get_metrics

        get_metrics().record_stream_cancel_orphan()
    except Exception:  # pragma: no cover - defensive
        pass


async def stream_with_backpressure(
    sync_iterator: Iterator[Any],
    format_item: Callable[[Any], str],
    format_done: Callable[[], str],
    request: "Request | None" = None,
    timeout: float | None = None,
    queue_size: int = DEFAULT_QUEUE_SIZE,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    format_heartbeat: Callable[[], str] | None = None,
) -> AsyncIterator[str]:
    """Stream items with backpressure, timeout, and disconnect detection.

    Args:
        sync_iterator: Synchronous iterator to stream from.
        format_item: Function to format each item as a string.
        format_done: Function to format the final "done" message.
        request: Optional Starlette Request for disconnect detection.
        timeout: Maximum time for the entire stream (seconds). ``None`` (the
            default for every route) resolves to ``config.generation_timeout``
            so a streamed request gets the SAME operator-configurable wall-clock
            budget as a non-streamed one — previously this was hardcoded to 300s
            and ignored the config, truncating long streamed generations that a
            non-streamed call would have allowed.
        queue_size: Maximum items in the backpressure queue.
        heartbeat_interval: Time between heartbeat messages when idle.
        format_heartbeat: Optional function to format heartbeat messages.

    Yields:
        Formatted string responses.

    Raises:
        StreamTimeoutError: If the stream exceeds the timeout.
        StreamCancelledError: If the client disconnects.
    """
    if timeout is None:
        timeout = _hfl_config.generation_timeout
    # Use bounded queue for backpressure
    queue: asyncio.Queue[Any | None | Exception] = asyncio.Queue(maxsize=queue_size)
    loop = asyncio.get_running_loop()
    start_time = time.monotonic()
    cancelled = asyncio.Event()

    def producer() -> None:
        """Run sync iterator and put items in queue."""
        try:
            for item in sync_iterator:
                if cancelled.is_set():
                    break
                # Use blocking put via run_coroutine_threadsafe for backpressure
                future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
                # Wait for queue space with configurable backpressure timeout.
                future.result(timeout=_hfl_config.stream_queue_put_timeout)
        except Exception as e:
            if not cancelled.is_set():
                logger.error("Error in stream producer: %s", e)
                loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            # Close the generator from the thread that drove it so the engine
            # runs its cooperative cancel / cleanup (CON-3) before the worker
            # exits, rather than leaking it until GC.
            _close_iterator(sync_iterator)
            loop.call_soon_threadsafe(queue.put_nowait, None)

    # Start producer in thread pool
    producer_task = asyncio.create_task(asyncio.to_thread(producer))

    completed_normally = False
    try:
        while True:
            # Check total timeout
            elapsed = time.monotonic() - start_time
            if elapsed > timeout:
                logger.warning("Stream timed out after %.1fs", elapsed)
                raise StreamTimeoutError(f"Stream exceeded {timeout}s timeout")

            # Check client disconnect
            if request is not None:
                try:
                    if await request.is_disconnected():
                        logger.info("Client disconnected, stopping stream")
                        raise StreamCancelledError("Client disconnected")
                except AttributeError:
                    # is_disconnected() may not be available
                    pass

            # Wait for item with heartbeat timeout
            try:
                item = await asyncio.wait_for(queue.get(), timeout=heartbeat_interval)
            except asyncio.TimeoutError:
                # No item received within heartbeat interval
                if format_heartbeat is not None:
                    yield format_heartbeat()
                continue

            if item is None:
                # Stream completed
                break
            if isinstance(item, Exception):
                raise item

            yield format_item(item)

        # The producer has delivered its sentinel (the ``None`` that broke the
        # loop), so the stream is logically complete: mark it BEFORE yielding
        # the done frame, otherwise a teardown at the done-yield boundary (or a
        # raising ``format_done``) would skip the flag and record a spurious
        # orphan.
        completed_normally = True
        # Send done message
        yield format_done()

    finally:
        # Signal the producer to stop; it re-checks the flag at the top of its
        # loop and then closes the generator (running the engine's cooperative
        # cancel).
        cancelled.set()
        if not producer_task.done():
            # Premature teardown (timeout / client disconnect) with the
            # producer still inside the engine's blocking next(): we cannot
            # preempt a sync engine call without an engine-level cancellation
            # API, so the worker may briefly outlive the stream. Record it for
            # observability (parity with the WS orphan counter).
            if not completed_normally:
                _record_stream_orphan()
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass


async def simple_stream_async(
    sync_iterator: Iterator[Any],
    format_item: Callable[[Any], str],
    format_done: Callable[[], str],
) -> AsyncIterator[str]:
    """Simple async streaming without backpressure.

    For simpler use cases where backpressure is not needed.
    Uses a basic queue to convert sync iterator to async.

    Args:
        sync_iterator: Synchronous iterator to stream from.
        format_item: Function to format each item.
        format_done: Function to format the done message.

    Yields:
        Formatted string responses.
    """
    # CON-6: bound the queue so a slow consumer applies backpressure instead
    # of the producer growing it without limit.
    queue: asyncio.Queue[Any | None | Exception] = asyncio.Queue(maxsize=DEFAULT_QUEUE_SIZE)
    loop = asyncio.get_running_loop()
    cancelled = asyncio.Event()

    def producer() -> None:
        try:
            for item in sync_iterator:
                if cancelled.is_set():
                    break
                # Blocking put for backpressure (bounded by the configured
                # put timeout) rather than an unbounded put_nowait.
                future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
                future.result(timeout=_hfl_config.stream_queue_put_timeout)
        except Exception as e:
            if not cancelled.is_set():
                logger.error("Error in stream producer: %s", e)
                loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            # Close the generator from the producer thread so the engine runs
            # its cooperative cancel / cleanup instead of leaking until GC.
            _close_iterator(sync_iterator)
            loop.call_soon_threadsafe(queue.put_nowait, None)

    producer_task = asyncio.create_task(asyncio.to_thread(producer))

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield format_item(item)

        yield format_done()
    finally:
        cancelled.set()
        if not producer_task.done():
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass
