# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Stress tests for concurrent streaming operations."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from hfl.engine.async_wrapper import AsyncEngineWrapper
from hfl.engine.base import ChatMessage


def create_streaming_engine(tokens: list[str] | None = None) -> MagicMock:
    """Create a mock engine that streams tokens.

    Args:
        tokens: Tokens to stream. Defaults to ["Hello", " ", "world", "!"].

    Returns:
        A MagicMock configured as a streaming InferenceEngine.
    """
    if tokens is None:
        tokens = ["Hello", " ", "world", "!"]

    engine = MagicMock()
    engine.is_loaded = True
    engine.model_name = "test-model"
    engine.generate_stream.return_value = iter(tokens)
    engine.chat_stream.return_value = iter(tokens)
    engine.generate.return_value = MagicMock(
        text="Hello world!", tokens_generated=4, stop_reason="stop"
    )
    engine.chat.return_value = MagicMock(
        text="Hello world!", tokens_generated=4, stop_reason="stop"
    )
    return engine


@pytest.mark.slow
class TestConcurrentStreaming:
    """Tests for concurrent streaming operations via AsyncEngineWrapper."""

    @pytest.mark.asyncio
    async def test_single_stream(self) -> None:
        """Single stream collects all tokens."""
        engine = create_streaming_engine()
        wrapper = AsyncEngineWrapper(engine)

        tokens: list[str] = []
        async for token in wrapper.generate_stream("test"):
            tokens.append(token)

        assert tokens == ["Hello", " ", "world", "!"]

    @pytest.mark.asyncio
    async def test_concurrent_streams(self) -> None:
        """Multiple concurrent streams each get their own tokens."""

        async def run_stream(stream_id: int) -> list[str]:
            tokens_list = [f"token{stream_id}_{j}" for j in range(5)]
            engine = create_streaming_engine(tokens_list)
            wrapper = AsyncEngineWrapper(engine)

            collected: list[str] = []
            async for token in wrapper.generate_stream(f"prompt-{stream_id}"):
                collected.append(token)
            return collected

        tasks = [run_stream(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        for i, tokens in enumerate(results):
            assert len(tokens) == 5
            assert tokens[0] == f"token{i}_0"

    @pytest.mark.asyncio
    async def test_concurrent_chat_streams(self) -> None:
        """Multiple concurrent chat streams work."""

        async def run_chat_stream(stream_id: int) -> list[str]:
            tokens_list = [f"chat{stream_id}"]
            engine = create_streaming_engine(tokens_list)
            wrapper = AsyncEngineWrapper(engine)

            msgs = [ChatMessage(role="user", content=f"msg-{stream_id}")]
            collected: list[str] = []
            async for token in wrapper.chat_stream(msgs):
                collected.append(token)
            return collected

        tasks = [run_chat_stream(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_stream_error_isolation(self) -> None:
        """Error in one stream doesn't affect others."""

        async def run_ok_stream() -> list[str]:
            engine = create_streaming_engine(["ok"])
            wrapper = AsyncEngineWrapper(engine)
            tokens: list[str] = []
            async for t in wrapper.generate_stream("test"):
                tokens.append(t)
            return tokens

        async def run_error_stream() -> list[str]:
            engine = MagicMock()
            engine.is_loaded = True

            def error_gen(*args, **kwargs):
                raise RuntimeError("Stream error")

            engine.generate_stream.side_effect = error_gen
            wrapper = AsyncEngineWrapper(engine)

            collected: list[str] = []
            async for token in wrapper.generate_stream("test"):
                collected.append(token)
            return collected  # Should never reach here

        # Run mix of good and bad streams concurrently
        ok_task = asyncio.create_task(run_ok_stream())
        error_task = asyncio.create_task(run_error_stream())

        # The good stream should complete successfully
        ok_result = await ok_task
        assert ok_result == ["ok"]

        # The error stream should have raised RuntimeError
        with pytest.raises(RuntimeError, match="Stream error"):
            await error_task

    @pytest.mark.asyncio
    async def test_concurrent_generate_non_streaming(self) -> None:
        """Concurrent non-streaming generates work."""

        async def run_generate(idx: int):
            engine = create_streaming_engine()
            wrapper = AsyncEngineWrapper(engine)
            return await wrapper.generate(f"prompt-{idx}")

        tasks = [run_generate(i) for i in range(20)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 20
        assert all(r.text == "Hello world!" for r in results)
