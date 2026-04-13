# SPDX-License-Identifier: HRUL-1.0
"""End-to-end concurrency contract tests (spec §5.3).

These tests drive the real FastAPI app with an ``httpx.AsyncClient`` and a
programmable fake engine so they can reproduce the exact scenarios the
spec demands without needing real model weights:

- Parallel requests are serialized when ``max_inflight`` = 1
- Wait queue overflow produces structured 429 ``QUEUE_FULL`` responses
- Slot-acquire timeout produces structured 503 ``QUEUE_TIMEOUT``
- ``/healthz`` reports live ``queue_depth`` / ``queue_in_flight``
- ``X-Queue-Depth`` headers appear on every response
- Streaming responses hold the slot for the entire generator
- An exception in the engine still releases the slot (no leak)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Iterator
from unittest.mock import MagicMock

import httpx
import pytest

from hfl.api.middleware import reset_rate_limiter
from hfl.api.server import app
from hfl.api.state import get_state
from hfl.core import get_container
from hfl.engine.base import ChatMessage, GenerationConfig, GenerationResult
from hfl.engine.dispatcher import InferenceDispatcher

# --- Fake engine --------------------------------------------------------------


@dataclass
class FakeSlowEngine:
    """Engine double that can block for a configurable duration.

    Each call sleeps for ``delay_seconds`` then returns a fixed string.
    ``call_count`` records how many times the engine was invoked.
    """

    delay_seconds: float = 0.05
    raise_on_call: bool = False
    call_count: int = 0
    concurrent_calls: int = 0
    max_concurrent_observed: int = 0
    is_loaded: bool = True
    _lock: Any = field(default=None)

    def __post_init__(self) -> None:
        import threading

        self._lock = threading.Lock()

    def _enter(self) -> None:
        with self._lock:
            self.concurrent_calls += 1
            self.call_count += 1
            if self.concurrent_calls > self.max_concurrent_observed:
                self.max_concurrent_observed = self.concurrent_calls

    def _exit(self) -> None:
        with self._lock:
            self.concurrent_calls -= 1

    def chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> GenerationResult:
        self._enter()
        try:
            time.sleep(self.delay_seconds)
            if self.raise_on_call:
                raise RuntimeError("boom")
            return GenerationResult(text="ok", tokens_generated=1)
        finally:
            self._exit()

    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> Iterator[str]:
        self._enter()
        try:
            # Emit a single token after the delay.
            time.sleep(self.delay_seconds)
            yield "ok"
        finally:
            self._exit()

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        self._enter()
        try:
            time.sleep(self.delay_seconds)
            return GenerationResult(text="ok", tokens_generated=1)
        finally:
            self._exit()

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        self._enter()
        try:
            time.sleep(self.delay_seconds)
            yield "ok"
        finally:
            self._exit()


# --- Fixtures -----------------------------------------------------------------


def _install_fake_engine(delay: float = 0.05, raise_on_call: bool = False) -> FakeSlowEngine:
    eng = FakeSlowEngine(delay_seconds=delay, raise_on_call=raise_on_call)
    mock_model = MagicMock()
    mock_model.name = "qwen3-32b-q4_k_m"
    get_state().engine = eng
    get_state().current_model = mock_model
    return eng


def _install_dispatcher(
    *, max_inflight: int = 1, max_queued: int = 16, acquire_timeout: float = 60.0
) -> InferenceDispatcher:
    """Replace the container's dispatcher singleton with a fresh one.

    We go through the container so subsequent ``get_dispatcher()`` calls
    (including inside route handlers and middleware) see the same
    instance.
    """
    c = get_container()
    c.dispatcher.reset()
    d = InferenceDispatcher(
        max_inflight=max_inflight,
        max_queued=max_queued,
        acquire_timeout=acquire_timeout,
    )
    # Monkeypatch the singleton cache so the factory is never called.
    c.dispatcher._instance = d  # type: ignore[attr-defined]
    return d


@pytest.fixture(autouse=True)
def _reset_state_and_dispatcher():
    get_state().api_key = None
    get_state().engine = None
    get_state().current_model = None
    reset_rate_limiter()
    yield
    get_state().api_key = None
    get_state().engine = None
    get_state().current_model = None
    reset_rate_limiter()
    # Restore dispatcher to config defaults so we don't leak test state.
    c = get_container()
    c.dispatcher.reset()


@pytest.fixture
async def aclient():
    # ``raise_app_exceptions=False`` lets the transport turn uncaught
    # route-handler exceptions into 500 responses, mirroring the real
    # Uvicorn behaviour and letting us assert on status codes.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# --- 1. Serialization --------------------------------------------------------


class TestSerialization:
    async def test_three_parallel_chat_requests_are_serialized(self, aclient):
        """With ``max_inflight=1`` and a 100 ms engine, three parallel
        calls must take ~300 ms (not 100 ms), and the engine must never
        observe more than one concurrent caller."""
        eng = _install_fake_engine(delay=0.1)
        _install_dispatcher(max_inflight=1, max_queued=8)

        async def _call():
            return await aclient.post(
                "/api/chat",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "stream": False,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        t0 = time.perf_counter()
        responses = await asyncio.gather(_call(), _call(), _call())
        elapsed = time.perf_counter() - t0

        assert all(r.status_code == 200 for r in responses)
        assert eng.call_count == 3
        assert eng.max_concurrent_observed == 1
        assert elapsed >= 0.28, f"requests were not serialized (elapsed={elapsed:.2f}s)"

    async def test_max_inflight_two_allows_two_in_parallel(self, aclient):
        eng = _install_fake_engine(delay=0.1)
        _install_dispatcher(max_inflight=2, max_queued=8)

        async def _call():
            return await aclient.post(
                "/api/chat",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "stream": False,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        t0 = time.perf_counter()
        responses = await asyncio.gather(*(_call() for _ in range(4)))
        elapsed = time.perf_counter() - t0

        assert all(r.status_code == 200 for r in responses)
        assert eng.max_concurrent_observed == 2
        # ~0.2 s with two-wide pipeline (2 batches of 2).
        assert elapsed < 0.35


# --- 2. Queue full -----------------------------------------------------------


class TestQueueFull:
    async def test_queue_full_returns_structured_429(self, aclient):
        """With max_inflight=1 and max_queued=1, three parallel requests
        cause the third to be rejected with ``QUEUE_FULL``."""
        _install_fake_engine(delay=0.2)
        _install_dispatcher(max_inflight=1, max_queued=1)

        async def _call():
            return await aclient.post(
                "/api/chat",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "stream": False,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        responses = await asyncio.gather(_call(), _call(), _call(), return_exceptions=True)
        statuses = sorted(r.status_code for r in responses)
        assert statuses == [200, 200, 429], statuses

        rejected = next(r for r in responses if r.status_code == 429)
        body = rejected.json()
        # Factory-style flat envelope (ErrorDetail.model_dump()).
        assert body["code"] == "QUEUE_FULL"
        assert body["category"] == "rate_limit"
        assert body["retryable"] is True
        assert "retry_after_seconds" in body["details"]
        # Required Ollama-style headers
        assert "retry-after" in {k.lower() for k in rejected.headers.keys()}
        assert "x-queue-depth" in {k.lower() for k in rejected.headers.keys()}


# --- 3. Queue acquire timeout ------------------------------------------------


class TestQueueTimeout:
    async def test_slot_acquire_times_out_to_503(self, aclient):
        """A second caller with a very small acquire timeout must see a
        ``QUEUE_TIMEOUT`` 503 when the first call takes longer than the
        acquire cap."""
        _install_fake_engine(delay=0.3)
        _install_dispatcher(max_inflight=1, max_queued=4, acquire_timeout=0.1)

        async def _call():
            return await aclient.post(
                "/api/chat",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "stream": False,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        first, second = await asyncio.gather(_call(), _call())

        # Exactly one succeeds; the other hit the acquire timeout.
        codes = sorted([first.status_code, second.status_code])
        assert codes == [200, 503], codes

        rejected = first if first.status_code == 503 else second
        body = rejected.json()
        assert body["code"] == "QUEUE_TIMEOUT"
        assert body["category"] == "engine"
        assert body["retryable"] is True


# --- 4. Exception still releases slot ----------------------------------------


class TestExceptionSafety:
    async def test_engine_exception_releases_slot(self, aclient):
        """If the engine raises, the dispatcher slot must still be
        released so the next caller can proceed."""
        _install_fake_engine(delay=0.02, raise_on_call=True)
        d = _install_dispatcher(max_inflight=1, max_queued=4)

        resp1 = await aclient.post(
            "/api/chat",
            json={
                "model": "qwen3-32b-q4_k_m",
                "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp1.status_code >= 500  # engine raised
        assert d.in_flight == 0
        assert d.depth == 0

        # A follow-up request must not hang — swap in a working engine.
        _install_fake_engine(delay=0.02, raise_on_call=False)
        resp2 = await aclient.post(
            "/api/chat",
            json={
                "model": "qwen3-32b-q4_k_m",
                "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp2.status_code == 200


# --- 5. /healthz live depth ---------------------------------------------------


class TestHealthzLive:
    async def test_healthz_reports_live_in_flight(self, aclient):
        _install_fake_engine(delay=0.3)
        _install_dispatcher(max_inflight=1, max_queued=4)

        async def _do_chat():
            return await aclient.post(
                "/api/chat",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "stream": False,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        task = asyncio.create_task(_do_chat())
        # Let the chat begin.
        await asyncio.sleep(0.05)
        probe = await aclient.get("/healthz")
        assert probe.status_code == 200
        body = probe.json()
        assert body["queue_in_flight"] >= 1
        assert "queue_depth" in body

        resp = await task
        assert resp.status_code == 200


# --- 6. Observability header --------------------------------------------------


class TestObservabilityHeaders:
    async def test_x_queue_headers_on_every_response(self, aclient):
        _install_dispatcher(max_inflight=2, max_queued=4)
        resp = await aclient.get("/api/version")
        assert resp.status_code == 200
        hdrs = {k.lower(): v for k, v in resp.headers.items()}
        for h in (
            "x-queue-depth",
            "x-queue-in-flight",
            "x-queue-max-inflight",
            "x-queue-max-size",
        ):
            assert h in hdrs, f"missing header {h}"
        assert hdrs["x-queue-max-inflight"] == "2"
        assert hdrs["x-queue-max-size"] == "4"


# --- 7. Streaming holds the slot ----------------------------------------------


class TestStreamingHoldsSlot:
    async def test_stream_holds_slot_until_complete(self, aclient):
        """A streaming request must hold its slot for the whole
        duration; a second stream request starts only once the first
        finishes."""
        eng = _install_fake_engine(delay=0.15)
        _install_dispatcher(max_inflight=1, max_queued=4)

        async def _stream_once():
            async with aclient.stream(
                "POST",
                "/api/chat",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            ) as resp:
                assert resp.status_code == 200
                async for _ in resp.aiter_lines():
                    pass

        t0 = time.perf_counter()
        await asyncio.gather(_stream_once(), _stream_once())
        elapsed = time.perf_counter() - t0

        assert eng.max_concurrent_observed == 1
        assert elapsed >= 0.25

    async def test_stream_queue_full_returns_429(self, aclient):
        """Second stream request is queued, third is rejected."""
        _install_fake_engine(delay=0.3)
        _install_dispatcher(max_inflight=1, max_queued=1)

        async def _stream_once():
            try:
                async with aclient.stream(
                    "POST",
                    "/api/chat",
                    json={
                        "model": "qwen3-32b-q4_k_m",
                        "stream": True,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                ) as resp:
                    status = resp.status_code
                    if status == 200:
                        async for _ in resp.aiter_lines():
                            pass
                    return status
            except httpx.HTTPError as e:
                return f"err:{e}"

        # Fire 3 streams in parallel. With 1 in-flight + 1 queued slot,
        # the third must be rejected.
        results = await asyncio.gather(_stream_once(), _stream_once(), _stream_once())
        statuses = sorted([r for r in results if isinstance(r, int)])
        assert statuses.count(200) == 2
        assert statuses.count(429) == 1


# --- 8. /api/generate also serialized -----------------------------------------


class TestGenerateSerialized:
    async def test_generate_is_serialized(self, aclient):
        eng = _install_fake_engine(delay=0.1)
        _install_dispatcher(max_inflight=1, max_queued=4)

        async def _gen():
            return await aclient.post(
                "/api/generate",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "prompt": "hello",
                    "stream": False,
                },
            )

        responses = await asyncio.gather(_gen(), _gen(), _gen())
        assert all(r.status_code == 200 for r in responses)
        assert eng.max_concurrent_observed == 1


# --- 9. Mixed endpoints share the dispatcher ---------------------------------


class TestCrossEndpointSharing:
    async def test_ollama_openai_anthropic_share_dispatcher(self, aclient):
        """A single dispatcher serves all three API surfaces; a slow
        call on one must block a call on another."""
        eng = _install_fake_engine(delay=0.1)
        _install_dispatcher(max_inflight=1, max_queued=8)

        async def _ollama():
            return await aclient.post(
                "/api/chat",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "stream": False,
                    "messages": [{"role": "user", "content": "a"}],
                },
            )

        async def _openai():
            return await aclient.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "stream": False,
                    "messages": [{"role": "user", "content": "b"}],
                },
            )

        async def _anthropic():
            return await aclient.post(
                "/v1/messages",
                json={
                    "model": "qwen3-32b-q4_k_m",
                    "stream": False,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "c"}],
                },
            )

        t0 = time.perf_counter()
        r1, r2, r3 = await asyncio.gather(_ollama(), _openai(), _anthropic())
        elapsed = time.perf_counter() - t0
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 200
        assert eng.max_concurrent_observed == 1
        assert elapsed >= 0.28
