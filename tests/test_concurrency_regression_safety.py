# SPDX-License-Identifier: HRUL-1.0
"""Regression / canary suite for the inference dispatcher (spec §5.3).

These tests are the safety net: after all the dispatcher changes they
must remain green forever. They prove the end-to-end pipeline still
functions under realistic concurrent pressure and they would catch any
future regression that accidentally leaks dispatcher state.

Scenarios:

- **Soak**: many parallel requests against a bounded dispatcher — all
  accepted or all queued, no depth/in_flight leak at the end.
- **Exception recovery**: alternating failing and succeeding calls;
  dispatcher counters must be consistent at the end.
- **Stability**: 50 request pairs in sequence; snapshot shows no
  accumulated leak.
- **Mixed endpoints**: rotating /api/chat, /api/generate, and
  /v1/chat/completions; the one shared dispatcher remains consistent.
- **Full recovery after queue full**: after rejections, new requests
  succeed once the in-flight call drains.
"""

from __future__ import annotations

import asyncio
import time
from typing import Iterator
from unittest.mock import MagicMock

import httpx
import pytest

from hfl.api.middleware import reset_rate_limiter
from hfl.api.server import app
from hfl.api.state import get_state
from hfl.core import get_container
from hfl.engine.base import ChatMessage, GenerationConfig, GenerationResult
from hfl.engine.dispatcher import InferenceDispatcher

pytestmark = pytest.mark.acceptance


class _Counter:
    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self._n = 0
        self._cur = 0
        self._max = 0

    def enter(self) -> None:
        with self._lock:
            self._cur += 1
            self._n += 1
            self._max = max(self._max, self._cur)

    def exit(self) -> None:
        with self._lock:
            self._cur -= 1

    @property
    def total(self) -> int:
        return self._n

    @property
    def peak(self) -> int:
        return self._max


class _FakeEngine:
    """Minimal engine that takes ``delay`` seconds per call.

    Two modes: ``ok`` always succeeds, ``alternating`` raises every
    other call so we can test recovery.
    """

    is_loaded = True

    def __init__(self, delay: float = 0.02, mode: str = "ok") -> None:
        self.delay = delay
        self.mode = mode
        self.counter = _Counter()
        self._call_idx = 0
        import threading

        self._idx_lock = threading.Lock()

    def _should_raise(self) -> bool:
        if self.mode != "alternating":
            return False
        with self._idx_lock:
            idx = self._call_idx
            self._call_idx += 1
        return idx % 2 == 0

    def chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> GenerationResult:
        self.counter.enter()
        try:
            time.sleep(self.delay)
            if self._should_raise():
                raise RuntimeError("boom")
            return GenerationResult(text="ok", tokens_generated=1)
        finally:
            self.counter.exit()

    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> Iterator[str]:
        self.counter.enter()
        try:
            time.sleep(self.delay)
            yield "ok"
        finally:
            self.counter.exit()

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        self.counter.enter()
        try:
            time.sleep(self.delay)
            return GenerationResult(text="ok", tokens_generated=1)
        finally:
            self.counter.exit()

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        self.counter.enter()
        try:
            time.sleep(self.delay)
            yield "ok"
        finally:
            self.counter.exit()


def _install(
    delay: float = 0.02,
    mode: str = "ok",
    *,
    max_inflight: int = 2,
    max_queued: int = 32,
    acquire_timeout: float = 30.0,
) -> tuple[_FakeEngine, InferenceDispatcher]:
    eng = _FakeEngine(delay=delay, mode=mode)
    mock_model = MagicMock()
    mock_model.name = "qwen3-32b-q4_k_m"
    get_state().engine = eng
    get_state().current_model = mock_model

    c = get_container()
    c.dispatcher.reset()
    d = InferenceDispatcher(
        max_inflight=max_inflight,
        max_queued=max_queued,
        acquire_timeout=acquire_timeout,
    )
    c.dispatcher._instance = d  # type: ignore[attr-defined]
    return eng, d


@pytest.fixture(autouse=True)
def _reset():
    get_state().api_key = None
    get_state().engine = None
    get_state().current_model = None
    reset_rate_limiter()
    yield
    get_state().api_key = None
    get_state().engine = None
    get_state().current_model = None
    reset_rate_limiter()
    get_container().dispatcher.reset()


@pytest.fixture
async def aclient():
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# --- 1. Soak -----------------------------------------------------------------


class TestSoak:
    async def test_twenty_parallel_requests_all_succeed(self, aclient):
        eng, d = _install(delay=0.01, max_inflight=2, max_queued=32)

        async def _call(i: int):
            return await aclient.post(
                "/api/chat",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "stream": False,
                    "messages": [{"role": "user", "content": f"hi {i}"}],
                },
            )

        results = await asyncio.gather(*(_call(i) for i in range(20)))
        assert all(r.status_code == 200 for r in results)
        assert eng.counter.total == 20
        assert eng.counter.peak <= 2

        snap = d.snapshot()
        assert snap.in_flight == 0
        assert snap.depth == 0
        assert snap.accepted_total == 20
        assert snap.rejected_full_total == 0
        assert snap.rejected_timeout_total == 0


# --- 2. Exception recovery ---------------------------------------------------


class TestExceptionRecovery:
    async def test_alternating_failures_do_not_leak_slots(self, aclient):
        eng, d = _install(delay=0.01, mode="alternating", max_inflight=2, max_queued=32)

        async def _call(i: int):
            return await aclient.post(
                "/api/chat",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "stream": False,
                    "messages": [{"role": "user", "content": f"hi {i}"}],
                },
            )

        results = await asyncio.gather(*(_call(i) for i in range(10)))
        codes = sorted(r.status_code for r in results)
        # Roughly half fail, half succeed — exact split depends on
        # which slot each request claimed.
        assert all(c in (200, 500) for c in codes)
        assert any(c == 500 for c in codes)
        assert any(c == 200 for c in codes)

        snap = d.snapshot()
        assert snap.in_flight == 0, "in_flight leaked after exceptions"
        assert snap.depth == 0, "depth leaked after exceptions"
        assert snap.accepted_total == 10
        assert snap.rejected_full_total == 0
        assert snap.rejected_timeout_total == 0


# --- 3. Stability over many iterations ---------------------------------------


class TestStability:
    async def test_fifty_request_pairs_no_leak(self, aclient):
        eng, d = _install(delay=0.002, max_inflight=2, max_queued=8)

        async def _pair():
            r1 = aclient.post(
                "/api/chat",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "stream": False,
                    "messages": [{"role": "user", "content": "a"}],
                },
            )
            r2 = aclient.post(
                "/api/chat",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "stream": False,
                    "messages": [{"role": "user", "content": "b"}],
                },
            )
            return await asyncio.gather(r1, r2)

        for i in range(50):
            # The global rate limiter (60 req/window) would otherwise
            # kick in halfway through; the dispatcher is what we want
            # to exercise here so we keep it out of the way.
            reset_rate_limiter()
            results = await _pair()
            assert all(r.status_code == 200 for r in results), (
                f"iteration {i}: statuses {[r.status_code for r in results]}"
            )

        snap = d.snapshot()
        assert snap.in_flight == 0
        assert snap.depth == 0
        assert snap.rejected_full_total == 0


# --- 4. Mixed endpoints ------------------------------------------------------


class TestMixedEndpoints:
    async def test_mixed_endpoints_share_dispatcher(self, aclient):
        eng, d = _install(delay=0.01, max_inflight=2, max_queued=32)

        async def _ollama(i):
            return await aclient.post(
                "/api/chat",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "stream": False,
                    "messages": [{"role": "user", "content": f"o{i}"}],
                },
            )

        async def _openai(i):
            return await aclient.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "stream": False,
                    "messages": [{"role": "user", "content": f"p{i}"}],
                },
            )

        async def _anthropic(i):
            return await aclient.post(
                "/v1/messages",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "stream": False,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"a{i}"}],
                },
            )

        async def _generate(i):
            return await aclient.post(
                "/api/generate",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "prompt": f"g{i}",
                    "stream": False,
                },
            )

        coros = []
        for i in range(6):
            coros.append(_ollama(i))
            coros.append(_openai(i))
            coros.append(_anthropic(i))
            coros.append(_generate(i))

        results = await asyncio.gather(*coros)
        assert all(r.status_code == 200 for r in results)
        assert eng.counter.total == 24
        assert eng.counter.peak <= 2

        snap = d.snapshot()
        assert snap.in_flight == 0
        assert snap.depth == 0
        assert snap.accepted_total == 24


# --- 5. Recovery after queue full --------------------------------------------


class TestRecoveryAfterFull:
    async def test_dispatch_resumes_after_queue_full(self, aclient):
        """Once the in-flight call finishes and the queue drains, new
        requests must succeed again — a full queue is a transient
        condition, not a permanent failure."""
        eng, d = _install(delay=0.15, max_inflight=1, max_queued=1)

        async def _call():
            return await aclient.post(
                "/api/chat",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "stream": False,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        # Saturate: 3 in parallel → 2 succeed, 1 rejected.
        first_batch = await asyncio.gather(_call(), _call(), _call())
        statuses = sorted(r.status_code for r in first_batch)
        assert statuses == [200, 200, 429], statuses

        # Now the queue is empty; follow-up requests succeed.
        second_batch = await asyncio.gather(_call(), _call())
        assert all(r.status_code == 200 for r in second_batch)

        snap = d.snapshot()
        assert snap.in_flight == 0
        assert snap.depth == 0
        assert snap.rejected_full_total == 1
        assert snap.accepted_total == 4
