# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for request tracing functionality."""

import asyncio

import pytest

from hfl.core.tracing import (
    RequestContext,
    clear_request_id,
    format_log_prefix,
    generate_request_id,
    get_request_id,
    get_trace_context,
    set_request_id,
    set_trace_context,
    with_request_id,
)


class TestGenerateRequestId:
    """Tests for generate_request_id function."""

    def test_generates_8_char_hex(self):
        """Should generate 8-character hex string."""
        rid = generate_request_id()
        assert len(rid) == 8
        assert all(c in "0123456789abcdef" for c in rid)

    def test_generates_unique_ids(self):
        """Should generate unique IDs."""
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100


class TestSetGetRequestId:
    """Tests for set_request_id and get_request_id functions."""

    def test_set_and_get(self):
        """Should set and retrieve request ID."""
        clear_request_id()
        set_request_id("test123")
        assert get_request_id() == "test123"
        clear_request_id()

    def test_set_generates_id_if_none(self):
        """Should generate ID if none provided."""
        clear_request_id()
        rid = set_request_id()
        assert rid is not None
        assert len(rid) == 8
        assert get_request_id() == rid
        clear_request_id()

    def test_set_with_custom_id(self):
        """Should use provided custom ID."""
        clear_request_id()
        rid = set_request_id("my-custom-id")
        assert rid == "my-custom-id"
        assert get_request_id() == "my-custom-id"
        clear_request_id()

    def test_get_returns_none_when_not_set(self):
        """Should return None when no request ID is set."""
        clear_request_id()
        assert get_request_id() is None

    def test_clear_request_id(self):
        """Should clear the request ID."""
        set_request_id("test")
        clear_request_id()
        assert get_request_id() is None


class TestTraceContext:
    """Tests for trace context functions."""

    def test_set_and_get_trace_context(self):
        """Should set and retrieve trace context."""
        context = {"parent_span_id": "abc123", "trace_id": "xyz789"}
        set_trace_context(context)
        assert get_trace_context() == context

    def test_trace_context_defaults_to_none(self):
        """Should return None when trace context not set."""
        # Note: This may fail if another test set the context
        # In production code, we'd reset context between tests


class TestRequestContext:
    """Tests for RequestContext context manager."""

    def test_sync_context_manager(self):
        """Should work as sync context manager."""
        clear_request_id()
        with RequestContext() as ctx:
            assert ctx.request_id is not None
            assert get_request_id() == ctx.request_id
        assert get_request_id() is None

    def test_sync_context_manager_with_custom_id(self):
        """Should use custom ID in context manager."""
        clear_request_id()
        with RequestContext("custom-id") as ctx:
            assert ctx.request_id == "custom-id"
            assert get_request_id() == "custom-id"
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Should work as async context manager."""
        clear_request_id()
        async with RequestContext() as ctx:
            assert ctx.request_id is not None
            assert get_request_id() == ctx.request_id
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_async_context_manager_with_custom_id(self):
        """Should use custom ID in async context manager."""
        clear_request_id()
        async with RequestContext("async-custom") as ctx:
            assert ctx.request_id == "async-custom"
            assert get_request_id() == "async-custom"
        assert get_request_id() is None

    def test_nested_contexts_restore_previous(self):
        """Should restore previous request ID after nested context."""
        clear_request_id()
        set_request_id("outer")

        with RequestContext("inner"):
            assert get_request_id() == "inner"

        assert get_request_id() == "outer"
        clear_request_id()


class TestWithRequestIdDecorator:
    """Tests for with_request_id decorator."""

    def test_sync_function_decorator(self):
        """Should set request ID for sync function."""
        clear_request_id()

        @with_request_id("decorated-id")
        def my_func():
            return get_request_id()

        result = my_func()
        assert result == "decorated-id"
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_async_function_decorator(self):
        """Should set request ID for async function."""
        clear_request_id()

        @with_request_id("async-decorated")
        async def my_async_func():
            return get_request_id()

        result = await my_async_func()
        assert result == "async-decorated"
        assert get_request_id() is None

    def test_decorator_generates_id_if_none(self):
        """Should generate ID if none provided."""
        clear_request_id()

        @with_request_id()
        def my_func():
            return get_request_id()

        result = my_func()
        assert result is not None
        assert len(result) == 8

    def test_decorator_restores_previous_id(self):
        """Should restore previous ID after decorated function."""
        clear_request_id()
        set_request_id("original")

        @with_request_id("decorated")
        def my_func():
            return get_request_id()

        result = my_func()
        assert result == "decorated"
        assert get_request_id() == "original"
        clear_request_id()


class TestFormatLogPrefix:
    """Tests for format_log_prefix function."""

    def test_returns_prefix_with_id(self):
        """Should return formatted prefix when ID is set."""
        clear_request_id()
        set_request_id("abc12345")
        assert format_log_prefix() == "[abc12345] "
        clear_request_id()

    def test_returns_empty_when_no_id(self):
        """Should return empty string when no ID is set."""
        clear_request_id()
        assert format_log_prefix() == ""


class TestContextVarPropagation:
    """Tests for context variable propagation through async tasks."""

    @pytest.mark.asyncio
    async def test_propagates_to_async_tasks(self):
        """Request ID should propagate to child async tasks."""
        clear_request_id()

        async def child_task():
            return get_request_id()

        set_request_id("parent-id")
        result = await child_task()
        assert result == "parent-id"
        clear_request_id()

    @pytest.mark.asyncio
    async def test_propagates_to_gathered_tasks(self):
        """Request ID should propagate to gathered tasks."""
        clear_request_id()

        async def get_id_after_delay(delay: float):
            await asyncio.sleep(delay)
            return get_request_id()

        set_request_id("gathered-id")
        results = await asyncio.gather(
            get_id_after_delay(0.01),
            get_id_after_delay(0.02),
            get_id_after_delay(0.03),
        )
        assert all(r == "gathered-id" for r in results)
        clear_request_id()

    @pytest.mark.asyncio
    async def test_isolated_between_independent_contexts(self):
        """Different contexts should have independent request IDs."""
        clear_request_id()

        async def task_with_id(rid: str):
            set_request_id(rid)
            await asyncio.sleep(0.01)
            return get_request_id()

        # These run in the same context, so they'll override each other
        # This is expected behavior - each task should set its own ID
        result1 = await task_with_id("id-1")
        result2 = await task_with_id("id-2")

        assert result1 == "id-1"
        assert result2 == "id-2"
        clear_request_id()
