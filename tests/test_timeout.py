# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for hfl.api.timeout – decorator and helper."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from hfl.api.errors import HFLHTTPException
from hfl.api.timeout import run_with_timeout, with_timeout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeConfig:
    generation_timeout: float = 600.0


# ---------------------------------------------------------------------------
# with_timeout decorator
# ---------------------------------------------------------------------------


class TestWithTimeoutDecorator:
    """Tests for the ``with_timeout`` decorator."""

    @pytest.mark.asyncio
    async def test_fast_function_completes(self) -> None:
        """A function that finishes before the deadline returns normally."""

        @with_timeout(5.0)
        async def fast() -> str:
            return "ok"

        assert await fast() == "ok"

    @pytest.mark.asyncio
    async def test_explicit_timeout_raises(self) -> None:
        """When the coroutine exceeds the explicit timeout, a 504 is raised."""

        @with_timeout(0.05)
        async def slow() -> None:
            await asyncio.sleep(10)

        with pytest.raises(HFLHTTPException) as exc_info:
            await slow()

        assert exc_info.value.status_code == 504

    @pytest.mark.asyncio
    async def test_default_timeout_from_config(self) -> None:
        """When timeout_seconds is None the config value is used."""
        fake_cfg = _FakeConfig(generation_timeout=0.05)

        @with_timeout()  # no explicit timeout
        async def slow() -> None:
            await asyncio.sleep(10)

        with patch("hfl.config.config", fake_cfg):
            with pytest.raises(HFLHTTPException) as exc_info:
                await slow()

        assert exc_info.value.status_code == 504

    @pytest.mark.asyncio
    async def test_explicit_zero_respected(self) -> None:
        """timeout_seconds=0 should be treated as 'no time' (immediate timeout),
        not fall back to config."""

        @with_timeout(0)  # 0 is falsy but explicit
        async def slow() -> None:
            await asyncio.sleep(10)

        # 0-second timeout should fire immediately
        with pytest.raises(HFLHTTPException):
            await slow()

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self) -> None:
        """functools.wraps should keep __name__ and __doc__."""

        @with_timeout(5.0)
        async def documented_endpoint() -> None:
            """My docstring."""

        assert documented_endpoint.__name__ == "documented_endpoint"
        assert documented_endpoint.__doc__ == "My docstring."

    @pytest.mark.asyncio
    async def test_passes_args_and_kwargs(self) -> None:
        """Arguments are forwarded to the wrapped function."""

        @with_timeout(5.0)
        async def add(a: int, b: int, *, extra: int = 0) -> int:
            return a + b + extra

        assert await add(1, 2, extra=10) == 13

    @pytest.mark.asyncio
    async def test_error_detail_contains_operation(self) -> None:
        """The 504 error detail should mention the function name."""

        @with_timeout(0.01)
        async def my_endpoint() -> None:
            await asyncio.sleep(10)

        with pytest.raises(HFLHTTPException) as exc_info:
            await my_endpoint()

        detail = exc_info.value.detail
        assert detail["code"] == "TIMEOUT"
        assert "my_endpoint" in detail["details"]["operation"]


# ---------------------------------------------------------------------------
# run_with_timeout helper
# ---------------------------------------------------------------------------


class TestRunWithTimeout:
    """Tests for the ``run_with_timeout`` standalone helper."""

    @pytest.mark.asyncio
    async def test_fast_coro_completes(self) -> None:
        async def fast() -> int:
            return 42

        result = await run_with_timeout(fast(), timeout_seconds=5.0)
        assert result == 42

    @pytest.mark.asyncio
    async def test_timeout_raises_504(self) -> None:
        async def slow() -> None:
            await asyncio.sleep(10)

        with pytest.raises(HFLHTTPException) as exc_info:
            await run_with_timeout(slow(), timeout_seconds=0.05, operation_name="inference")

        assert exc_info.value.status_code == 504
        assert "inference" in exc_info.value.detail["details"]["operation"]

    @pytest.mark.asyncio
    async def test_default_timeout_from_config(self) -> None:
        fake_cfg = _FakeConfig(generation_timeout=0.05)

        async def slow() -> None:
            await asyncio.sleep(10)

        with patch("hfl.config.config", fake_cfg):
            with pytest.raises(HFLHTTPException):
                await run_with_timeout(slow())

    @pytest.mark.asyncio
    async def test_operation_name_default(self) -> None:
        async def slow() -> None:
            await asyncio.sleep(10)

        with pytest.raises(HFLHTTPException) as exc_info:
            await run_with_timeout(slow(), timeout_seconds=0.01)

        assert exc_info.value.detail["details"]["operation"] == "operation"
