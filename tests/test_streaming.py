# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for streaming utilities with backpressure."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from hfl.api.streaming import (
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_QUEUE_SIZE,
    DEFAULT_STREAM_TIMEOUT,
    StreamCancelledError,
    StreamTimeoutError,
    simple_stream_async,
    stream_with_backpressure,
)


class TestDefaults:
    """Tests for default configuration."""

    def test_default_timeout(self):
        """Default timeout should be 5 minutes."""
        assert DEFAULT_STREAM_TIMEOUT == 300.0

    def test_default_queue_size(self):
        """Default queue size should be 100."""
        assert DEFAULT_QUEUE_SIZE == 100

    def test_default_heartbeat_interval(self):
        """Default heartbeat interval should be 30 seconds."""
        assert DEFAULT_HEARTBEAT_INTERVAL == 30.0


class TestStreamWithBackpressure:
    """Tests for stream_with_backpressure function."""

    @pytest.mark.asyncio
    async def test_basic_streaming(self):
        """Should stream items from sync iterator."""

        def sync_iter():
            yield "a"
            yield "b"
            yield "c"

        results = []
        async for item in stream_with_backpressure(
            sync_iter(),
            format_item=lambda x: f"item:{x}",
            format_done=lambda: "done",
        ):
            results.append(item)

        assert results == ["item:a", "item:b", "item:c", "done"]

    @pytest.mark.asyncio
    async def test_empty_iterator(self):
        """Should handle empty iterator."""

        def sync_iter():
            return
            yield  # Make it a generator

        results = []
        async for item in stream_with_backpressure(
            sync_iter(),
            format_item=lambda x: f"item:{x}",
            format_done=lambda: "done",
        ):
            results.append(item)

        assert results == ["done"]

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Should raise StreamTimeoutError on timeout."""

        def slow_iter():
            time.sleep(0.3)  # Sleep longer than timeout
            yield "a"

        with pytest.raises(StreamTimeoutError, match="exceeded"):
            async for _ in stream_with_backpressure(
                slow_iter(),
                format_item=lambda x: x,
                format_done=lambda: "done",
                timeout=0.05,
                heartbeat_interval=0.02,
            ):
                pass

    @pytest.mark.asyncio
    async def test_client_disconnect(self):
        """Should raise StreamCancelledError on client disconnect."""
        mock_request = MagicMock()
        mock_request.is_disconnected = AsyncMock(return_value=True)

        def sync_iter():
            yield "a"
            time.sleep(0.1)  # Give time for disconnect check
            yield "b"

        with pytest.raises(StreamCancelledError, match="disconnected"):
            async for _ in stream_with_backpressure(
                sync_iter(),
                format_item=lambda x: x,
                format_done=lambda: "done",
                request=mock_request,
                heartbeat_interval=0.01,
            ):
                pass

    @pytest.mark.asyncio
    async def test_heartbeat_emission(self):
        """Should emit heartbeats during idle periods."""

        def slow_iter():
            time.sleep(0.15)
            yield "a"

        heartbeats = []

        def format_heartbeat():
            heartbeats.append("heartbeat")
            return ": heartbeat\n\n"

        results = []
        async for item in stream_with_backpressure(
            slow_iter(),
            format_item=lambda x: x,
            format_done=lambda: "done",
            heartbeat_interval=0.05,
            format_heartbeat=format_heartbeat,
        ):
            results.append(item)

        # Should have at least one heartbeat
        assert len(heartbeats) >= 1

    @pytest.mark.asyncio
    async def test_backpressure_bounded_queue(self):
        """Should apply backpressure with bounded queue."""
        consumed = []

        def sync_iter():
            for i in range(10):
                yield i

        async for item in stream_with_backpressure(
            sync_iter(),
            format_item=lambda x: (consumed.append(x), str(x))[1],
            format_done=lambda: "done",
            queue_size=5,
        ):
            pass  # Just consume without delay

        assert len(consumed) == 10

    @pytest.mark.asyncio
    async def test_exception_in_producer(self):
        """Should propagate exceptions from producer."""

        def error_iter():
            yield "a"
            raise ValueError("Producer error")

        with pytest.raises(ValueError, match="Producer error"):
            async for _ in stream_with_backpressure(
                error_iter(),
                format_item=lambda x: x,
                format_done=lambda: "done",
            ):
                pass

    @pytest.mark.asyncio
    async def test_cleanup_on_error(self):
        """Should cleanup producer task on error."""

        def sync_iter():
            yield "a"
            time.sleep(0.5)  # Sleep to ensure task is running when timeout hits
            yield "b"

        try:
            async for _ in stream_with_backpressure(
                sync_iter(),
                format_item=lambda x: x,
                format_done=lambda: "done",
                timeout=0.05,
                heartbeat_interval=0.02,
            ):
                pass
        except StreamTimeoutError:
            pass

        # Give some time for cleanup
        await asyncio.sleep(0.05)
        # Test passes if no hanging tasks

    @pytest.mark.asyncio
    async def test_request_without_is_disconnected(self):
        """Should handle request without is_disconnected method."""
        mock_request = MagicMock(spec=[])  # No is_disconnected

        def sync_iter():
            yield "a"

        results = []
        async for item in stream_with_backpressure(
            sync_iter(),
            format_item=lambda x: x,
            format_done=lambda: "done",
            request=mock_request,
        ):
            results.append(item)

        assert "a" in results


class TestSimpleStreamAsync:
    """Tests for simple_stream_async function."""

    @pytest.mark.asyncio
    async def test_basic_streaming(self):
        """Should stream items from sync iterator."""

        def sync_iter():
            yield "a"
            yield "b"
            yield "c"

        results = []
        async for item in simple_stream_async(
            sync_iter(),
            format_item=lambda x: f"item:{x}",
            format_done=lambda: "done",
        ):
            results.append(item)

        assert results == ["item:a", "item:b", "item:c", "done"]

    @pytest.mark.asyncio
    async def test_empty_iterator(self):
        """Should handle empty iterator."""

        def sync_iter():
            return
            yield

        results = []
        async for item in simple_stream_async(
            sync_iter(),
            format_item=lambda x: x,
            format_done=lambda: "done",
        ):
            results.append(item)

        assert results == ["done"]

    @pytest.mark.asyncio
    async def test_exception_propagation(self):
        """Should propagate exceptions from producer."""

        def error_iter():
            yield "a"
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError, match="Test error"):
            async for _ in simple_stream_async(
                error_iter(),
                format_item=lambda x: x,
                format_done=lambda: "done",
            ):
                pass

    @pytest.mark.asyncio
    async def test_many_items(self):
        """Should handle many items efficiently."""

        def sync_iter():
            for i in range(1000):
                yield i

        count = 0
        async for item in simple_stream_async(
            sync_iter(),
            format_item=str,
            format_done=lambda: "done",
        ):
            if item != "done":
                count += 1

        assert count == 1000


class TestExceptionClasses:
    """Tests for exception classes."""

    def test_stream_timeout_error(self):
        """StreamTimeoutError should be an Exception."""
        error = StreamTimeoutError("test message")
        assert isinstance(error, Exception)
        assert str(error) == "test message"

    def test_stream_cancelled_error(self):
        """StreamCancelledError should be an Exception."""
        error = StreamCancelledError("cancelled")
        assert isinstance(error, Exception)
        assert str(error) == "cancelled"


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_generator_with_none_values(self):
        """Should handle None values in stream (not as end marker)."""
        # Note: None is used as end marker, so actual None values
        # need to be handled differently in the format function

        def sync_iter():
            yield "a"
            yield "b"

        results = []
        async for item in stream_with_backpressure(
            sync_iter(),
            format_item=lambda x: str(x),
            format_done=lambda: "done",
        ):
            results.append(item)

        assert results == ["a", "b", "done"]

    @pytest.mark.asyncio
    async def test_format_functions_called(self):
        """Format functions should be called correctly."""
        format_item_calls = []
        format_done_calls = []

        def format_item(x):
            format_item_calls.append(x)
            return f"formatted:{x}"

        def format_done():
            format_done_calls.append(True)
            return "END"

        def sync_iter():
            yield 1
            yield 2

        results = []
        async for item in stream_with_backpressure(
            sync_iter(),
            format_item=format_item,
            format_done=format_done,
        ):
            results.append(item)

        assert format_item_calls == [1, 2]
        assert len(format_done_calls) == 1
        assert results == ["formatted:1", "formatted:2", "END"]

    @pytest.mark.asyncio
    async def test_concurrent_streams(self):
        """Should handle multiple concurrent streams."""

        async def run_stream(stream_id: int):
            def sync_iter():
                for i in range(5):
                    yield f"{stream_id}:{i}"

            results = []
            async for item in simple_stream_async(
                sync_iter(),
                format_item=lambda x: x,
                format_done=lambda: "done",
            ):
                results.append(item)
            return results

        # Run 3 streams concurrently
        tasks = [run_stream(i) for i in range(3)]
        all_results = await asyncio.gather(*tasks)

        for results in all_results:
            assert len(results) == 6  # 5 items + done
            assert results[-1] == "done"

    @pytest.mark.asyncio
    async def test_custom_queue_size(self):
        """Should respect custom queue size."""
        small_queue_size = 2

        def sync_iter():
            for i in range(5):
                yield i

        results = []
        async for item in stream_with_backpressure(
            sync_iter(),
            format_item=str,
            format_done=lambda: "done",
            queue_size=small_queue_size,
        ):
            results.append(item)

        assert len(results) == 6  # 5 items + done
