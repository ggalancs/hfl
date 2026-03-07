# SPDX-License-Identifier: HRUL-1.0
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

if TYPE_CHECKING:
    from starlette.requests import Request

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_STREAM_TIMEOUT = 300.0  # 5 minutes
DEFAULT_QUEUE_SIZE = 100  # Max tokens in buffer
DEFAULT_HEARTBEAT_INTERVAL = 30.0  # Seconds between heartbeats


class StreamTimeoutError(Exception):
    """Raised when a stream operation times out."""

    pass


class StreamCancelledError(Exception):
    """Raised when a stream is cancelled (e.g., client disconnect)."""

    pass


async def stream_with_backpressure(
    sync_iterator: Iterator[Any],
    format_item: Callable[[Any], str],
    format_done: Callable[[], str],
    request: "Request | None" = None,
    timeout: float = DEFAULT_STREAM_TIMEOUT,
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
        timeout: Maximum time for the entire stream (seconds).
        queue_size: Maximum items in the backpressure queue.
        heartbeat_interval: Time between heartbeat messages when idle.
        format_heartbeat: Optional function to format heartbeat messages.

    Yields:
        Formatted string responses.

    Raises:
        StreamTimeoutError: If the stream exceeds the timeout.
        StreamCancelledError: If the client disconnects.
    """
    # Use bounded queue for backpressure
    queue: asyncio.Queue[Any | None | Exception] = asyncio.Queue(maxsize=queue_size)
    loop = asyncio.get_event_loop()
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
                future.result(timeout=60)  # Wait for queue space with timeout
        except Exception as e:
            if not cancelled.is_set():
                logger.error("Error in stream producer: %s", e)
                loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    # Start producer in thread pool
    producer_task = asyncio.create_task(asyncio.to_thread(producer))

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

        # Send done message
        yield format_done()

    finally:
        # Signal producer to stop
        cancelled.set()
        # Cancel and cleanup producer if still running
        if not producer_task.done():
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
    queue: asyncio.Queue[Any | None | Exception] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def producer() -> None:
        try:
            for item in sync_iterator:
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as e:
            logger.error("Error in stream producer: %s", e)
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
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
        if not producer_task.done():
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass
