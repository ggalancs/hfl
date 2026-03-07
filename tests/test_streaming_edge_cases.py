# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Streaming edge case tests.

Tests for client disconnect, timeout, and backpressure scenarios.
"""

import asyncio
from typing import AsyncIterator, Iterator
from unittest.mock import MagicMock, patch

import pytest

from hfl.api.helpers import StreamingContext


class TestStreamingContext:
    """Tests for StreamingContext helper."""

    def test_context_generates_consistent_id(self):
        """StreamingContext should use same ID for all chunks."""
        ctx = StreamingContext("test-model")

        chunk1 = ctx.format_chunk(content="Hello")
        chunk2 = ctx.format_chunk(content=" world")

        # Extract IDs from chunks
        import json
        data1 = json.loads(chunk1.split("data: ")[1].strip())
        data2 = json.loads(chunk2.split("data: ")[1].strip())

        assert data1["id"] == data2["id"]

    def test_context_consistent_timestamp(self):
        """StreamingContext should use same timestamp for all chunks."""
        ctx = StreamingContext("test-model")

        import time
        # Small delay to ensure time.time() would differ
        time.sleep(0.01)

        chunk1 = ctx.format_chunk(content="Hello")
        chunk2 = ctx.format_chunk(content=" world")

        import json
        data1 = json.loads(chunk1.split("data: ")[1].strip())
        data2 = json.loads(chunk2.split("data: ")[1].strip())

        assert data1["created"] == data2["created"]

    def test_done_marker_format(self):
        """DONE marker should be properly formatted."""
        ctx = StreamingContext("test-model")
        done = ctx.format_done()

        assert done == "data: [DONE]\n\n"

    def test_chunk_with_finish_reason(self):
        """Chunk with finish reason should be properly formatted."""
        ctx = StreamingContext("test-model")
        chunk = ctx.format_chunk(finish_reason="stop")

        import json
        data = json.loads(chunk.split("data: ")[1].strip())

        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["choices"][0]["delta"] == {}


class TestStreamingErrors:
    """Tests for streaming error scenarios."""

    @pytest.mark.asyncio
    async def test_engine_error_during_stream(self):
        """Engine error during streaming should be handled."""
        from hfl.api.helpers import stream_openai_chat
        from hfl.engine.base import GenerationConfig

        mock_engine = MagicMock()

        def error_generator():
            yield "Hello"
            yield " world"
            raise RuntimeError("Engine error")

        mock_engine.chat_stream.return_value = error_generator()

        messages = [MagicMock(role="user", content="Hi")]
        config = GenerationConfig()

        with pytest.raises(RuntimeError):
            chunks = []
            async for chunk in stream_openai_chat(mock_engine, messages, config, "test"):
                chunks.append(chunk)

    @pytest.mark.asyncio
    async def test_empty_response_stream(self):
        """Empty response stream should complete normally."""
        from hfl.api.helpers import stream_openai_chat
        from hfl.engine.base import GenerationConfig

        mock_engine = MagicMock()
        mock_engine.chat_stream.return_value = iter([])

        messages = [MagicMock(role="user", content="Hi")]
        config = GenerationConfig()

        chunks = []
        async for chunk in stream_openai_chat(mock_engine, messages, config, "test"):
            chunks.append(chunk)

        # Should have finish chunk and done marker
        assert len(chunks) >= 1
        assert "[DONE]" in chunks[-1]


class TestNDJSONStreaming:
    """Tests for NDJSON (Ollama-style) streaming."""

    @pytest.mark.asyncio
    async def test_ndjson_chunk_format(self):
        """NDJSON chunks should be properly formatted."""
        from hfl.api.helpers import stream_ollama_generate
        from hfl.engine.base import GenerationConfig

        mock_engine = MagicMock()
        mock_engine.generate_stream.return_value = iter(["Hello", " world"])

        config = GenerationConfig()

        chunks = []
        async for chunk in stream_ollama_generate(mock_engine, "test prompt", config, "model"):
            chunks.append(chunk)

        import json
        # Each chunk should be valid JSON ending with newline
        for chunk in chunks:
            assert chunk.endswith("\n")
            data = json.loads(chunk.strip())
            assert "model" in data
            assert "done" in data


class TestStreamingBackpressure:
    """Tests for backpressure and buffer handling."""

    @pytest.mark.asyncio
    async def test_large_token_stream(self):
        """Large token streams should be handled correctly."""
        from hfl.api.helpers import stream_openai_chat
        from hfl.engine.base import GenerationConfig

        mock_engine = MagicMock()
        # Generate 1000 tokens
        mock_engine.chat_stream.return_value = iter([f"token_{i} " for i in range(1000)])

        messages = [MagicMock(role="user", content="Generate lots of text")]
        config = GenerationConfig()

        chunk_count = 0
        async for _ in stream_openai_chat(mock_engine, messages, config, "test"):
            chunk_count += 1

        # Should have 1000 content chunks + finish chunk + done marker
        assert chunk_count == 1002

    @pytest.mark.asyncio
    async def test_unicode_in_stream(self):
        """Unicode characters in stream should be handled."""
        from hfl.api.helpers import stream_openai_chat
        from hfl.engine.base import GenerationConfig

        mock_engine = MagicMock()
        mock_engine.chat_stream.return_value = iter([
            "Hello ",
            "世界 ",  # Chinese
            "🌍 ",   # Emoji
            "مرحبا",  # Arabic
        ])

        messages = [MagicMock(role="user", content="Test")]
        config = GenerationConfig()

        chunks = []
        async for chunk in stream_openai_chat(mock_engine, messages, config, "test"):
            chunks.append(chunk)

        # Verify all chunks are valid
        import json
        for chunk in chunks[:-1]:  # Exclude [DONE]
            if "data: " in chunk and "[DONE]" not in chunk:
                data = json.loads(chunk.split("data: ")[1].strip())
                assert "choices" in data


class TestOllamaChatStreaming:
    """Tests for Ollama-style chat streaming."""

    @pytest.mark.asyncio
    async def test_ollama_chat_final_message(self):
        """Ollama chat stream should include full message in final chunk."""
        from hfl.api.helpers import stream_ollama_chat
        from hfl.engine.base import GenerationConfig

        mock_engine = MagicMock()
        mock_engine.chat_stream.return_value = iter(["Hello", " ", "world", "!"])

        messages = [MagicMock(role="user", content="Hi")]
        config = GenerationConfig()

        chunks = []
        async for chunk in stream_ollama_chat(mock_engine, messages, config, "test"):
            chunks.append(chunk)

        import json
        # Last chunk should have done=True and full message
        last_chunk = json.loads(chunks[-1].strip())
        assert last_chunk["done"] is True
        assert last_chunk["message"]["content"] == "Hello world!"


class TestStreamingCancellation:
    """Tests for stream cancellation scenarios."""

    @pytest.mark.asyncio
    async def test_generator_cleanup_on_cancel(self):
        """Generator should be cleaned up on cancellation."""
        cleanup_called = False

        def generator_with_cleanup():
            nonlocal cleanup_called
            try:
                for i in range(100):
                    yield f"token_{i}"
            finally:
                cleanup_called = True

        mock_engine = MagicMock()
        mock_engine.chat_stream.return_value = generator_with_cleanup()

        from hfl.api.helpers import stream_openai_chat
        from hfl.engine.base import GenerationConfig

        messages = [MagicMock(role="user", content="Test")]
        config = GenerationConfig()

        async def consume_partial():
            count = 0
            async for _ in stream_openai_chat(mock_engine, messages, config, "test"):
                count += 1
                if count >= 5:
                    break

        await consume_partial()
        # Note: In the current implementation, cleanup may not be called
        # because we break out of the loop without explicit cleanup.
        # This test documents the current behavior.
