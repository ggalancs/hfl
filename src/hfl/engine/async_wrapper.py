# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Async wrappers for synchronous inference engines.

This module provides async wrappers that allow synchronous engines to be
used in async contexts without blocking the event loop.

Usage:
    from hfl.engine.async_wrapper import AsyncEngineWrapper

    engine = LlamaCppEngine()
    async_engine = AsyncEngineWrapper(engine)

    # Load asynchronously
    await async_engine.load("/path/to/model")

    # Generate asynchronously
    result = await async_engine.chat(messages, config)

    # Stream asynchronously
    async for token in async_engine.chat_stream(messages, config):
        print(token)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from hfl.engine.base import ChatMessage, GenerationConfig, GenerationResult, InferenceEngine

logger = logging.getLogger(__name__)


class AsyncEngineWrapper:
    """Wraps a synchronous InferenceEngine for async usage.

    This wrapper uses asyncio.to_thread() to run blocking engine operations
    in a thread pool, preventing them from blocking the event loop.

    For streaming operations, it converts the synchronous generator to an
    async generator using a queue-based approach.

    Attributes:
        engine: The wrapped synchronous engine.
    """

    def __init__(self, engine: "InferenceEngine") -> None:
        """Initialize the wrapper.

        Args:
            engine: The synchronous engine to wrap.
        """
        self._engine = engine

    @property
    def engine(self) -> "InferenceEngine":
        """Get the wrapped engine."""
        return self._engine

    @property
    def is_loaded(self) -> bool:
        """Check if the engine has a model loaded."""
        return self._engine.is_loaded

    @property
    def model_name(self) -> str:
        """Get the name of the loaded model."""
        return self._engine.model_name

    async def load(self, model_path: str, **kwargs) -> None:
        """Load a model asynchronously.

        Args:
            model_path: Path to the model files.
            **kwargs: Additional arguments passed to the engine.
        """
        await asyncio.to_thread(self._engine.load, model_path, **kwargs)

    async def unload(self) -> None:
        """Unload the current model asynchronously."""
        await asyncio.to_thread(self._engine.unload)

    async def generate(
        self, prompt: str, config: "GenerationConfig | None" = None
    ) -> "GenerationResult":
        """Generate text completion asynchronously.

        Args:
            prompt: The input prompt.
            config: Generation configuration.

        Returns:
            GenerationResult with the generated text and metadata.
        """
        return await asyncio.to_thread(self._engine.generate, prompt, config)

    async def chat(
        self, messages: list["ChatMessage"], config: "GenerationConfig | None" = None
    ) -> "GenerationResult":
        """Generate chat completion asynchronously.

        Args:
            messages: List of chat messages.
            config: Generation configuration.

        Returns:
            GenerationResult with the response and metadata.
        """
        return await asyncio.to_thread(self._engine.chat, messages, config)

    async def generate_stream(
        self, prompt: str, config: "GenerationConfig | None" = None
    ) -> AsyncIterator[str]:
        """Stream text completion asynchronously.

        Converts the synchronous generator to an async generator using a
        queue-based approach.

        Args:
            prompt: The input prompt.
            config: Generation configuration.

        Yields:
            Generated tokens one at a time.

        Raises:
            Exception: Re-raises any exception from the producer thread.
        """
        queue: asyncio.Queue[str | None | Exception] = asyncio.Queue(maxsize=100)
        loop = asyncio.get_event_loop()

        def producer() -> None:
            """Run sync generator and put tokens in queue."""
            try:
                for token in self._engine.generate_stream(prompt, config):
                    asyncio.run_coroutine_threadsafe(queue.put(token), loop).result(timeout=60)
            except Exception as e:
                logger.error("Error in stream producer: %s", e)
                # Send exception to consumer instead of swallowing it
                asyncio.run_coroutine_threadsafe(queue.put(e), loop).result(timeout=60)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result(timeout=60)

        # Start producer in thread pool
        producer_task = asyncio.create_task(asyncio.to_thread(producer))

        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    producer_task.cancel()
                    raise TimeoutError("Token stream timed out")
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # Ensure producer task completes
            if not producer_task.done():
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass
            else:
                await producer_task

    async def chat_stream(
        self, messages: list["ChatMessage"], config: "GenerationConfig | None" = None
    ) -> AsyncIterator[str]:
        """Stream chat completion asynchronously.

        Converts the synchronous generator to an async generator using a
        queue-based approach.

        Args:
            messages: List of chat messages.
            config: Generation configuration.

        Yields:
            Generated tokens one at a time.

        Raises:
            Exception: Re-raises any exception from the producer thread.
        """
        queue: asyncio.Queue[str | None | Exception] = asyncio.Queue(maxsize=100)
        loop = asyncio.get_event_loop()

        def producer() -> None:
            """Run sync generator and put tokens in queue."""
            try:
                for token in self._engine.chat_stream(messages, config):
                    asyncio.run_coroutine_threadsafe(queue.put(token), loop).result(timeout=60)
            except Exception as e:
                logger.error("Error in chat stream producer: %s", e)
                # Send exception to consumer instead of swallowing it
                asyncio.run_coroutine_threadsafe(queue.put(e), loop).result(timeout=60)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result(timeout=60)

        # Start producer in thread pool
        producer_task = asyncio.create_task(asyncio.to_thread(producer))

        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    producer_task.cancel()
                    raise TimeoutError("Token stream timed out")
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # Ensure producer task completes
            if not producer_task.done():
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass
            else:
                await producer_task
