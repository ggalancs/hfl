# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A reply nobody will read stops generating when its client leaves.

Measured on 2026-11-01: Project Hub gave up on a card at its 300 s timeout
and Qwen3.8 went on decoding to 2048 tokens, 380 s, holding a slot the
other requests shared. Uvicorn never cancels a handler whose client
disconnects, and a reply that is not streamed checked nothing. A real
server and a real socket: the test client cannot disconnect.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from hfl.api.server import app
from hfl.api.state import get_state, reset_state
from hfl.engine.base import GenerationResult


def _serve() -> tuple[object, int]:
    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started
    return server, port


@pytest.fixture
def server(temp_config):
    reset_state()
    srv, port = _serve()
    yield port
    srv.should_exit = True
    reset_state()


def _engine(sample_manifest):
    """A model that generates until its request is cancelled (5 s at most)."""
    from unittest.mock import MagicMock

    from hfl.engine import cancel

    seen = {"started": threading.Event(), "cancelled_after": None}

    def chat(messages, config=None, tools=None):
        signal = cancel.current()
        seen["started"].set()
        t0 = time.monotonic()
        if signal is not None and signal.wait(5):
            seen["cancelled_after"] = time.monotonic() - t0
        return GenerationResult(text="late", tokens_generated=1, tokens_prompt=1)

    engine = MagicMock(is_loaded=True, supports_images=False)
    engine.chat = MagicMock(side_effect=chat)
    state = get_state()
    state.engine = engine
    state.current_model = sample_manifest
    return seen


def _post_and_leave(port: int, path: str, body: dict, seen: dict) -> None:
    payload = json.dumps(body).encode()
    with socket.create_connection(("127.0.0.1", port)) as conn:
        conn.sendall(
            f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )
        assert seen["started"].wait(5), "the generation never started"
        time.sleep(0.2)
    # The socket is closed: the client has gone.


@pytest.mark.parametrize(
    ("path", "extra"),
    [
        ("/v1/chat/completions", {}),
        ("/api/chat", {"stream": False}),
    ],
)
def test_the_generation_stops_when_the_client_leaves(server, sample_manifest, path, extra):
    seen = _engine(sample_manifest)
    body = {
        "model": sample_manifest.name,
        "messages": [{"role": "user", "content": "write a long card"}],
        **extra,
    }
    _post_and_leave(server, path, body, seen)
    deadline = time.monotonic() + 4
    while seen["cancelled_after"] is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert seen["cancelled_after"] is not None, "it generated on for nobody"
    assert seen["cancelled_after"] < 2


def test_a_client_that_waits_gets_its_answer(server, sample_manifest):
    """The watch must not cut a client that is still there."""
    import httpx

    from hfl.engine import cancel

    def chat(messages, config=None, tools=None):
        signal = cancel.current()
        assert signal is not None and not signal.wait(0.5)  # half a second, uncancelled
        return GenerationResult(text="Paris.", tokens_generated=2, tokens_prompt=5)

    seen = _engine(sample_manifest)
    get_state().engine.chat.side_effect = chat
    reply = httpx.post(
        f"http://127.0.0.1:{server}/v1/chat/completions",
        json={"model": sample_manifest.name, "messages": [{"role": "user", "content": "q"}]},
        timeout=10,
    )
    assert reply.status_code == 200 and reply.json()["choices"][0]["message"]["content"] == "Paris."
    assert seen["cancelled_after"] is None


def test_the_timeout_still_fires_inside_a_watched_request():
    """With the client still there, HFL_GENERATION_TIMEOUT answers 504 and
    stops the engine, as before the watch."""
    import asyncio

    from fastapi import HTTPException

    from hfl.api import middleware
    from hfl.api.helpers import run_dispatched
    from hfl.engine import cancel

    stopped = threading.Event()

    def slow() -> str:
        signal = cancel.current()
        if signal is not None and signal.wait(5):
            stopped.set()
        return "late"

    async def call() -> None:
        token = middleware._client_gone.set(asyncio.Event())  # a client that stays
        try:
            await run_dispatched(slow, timeout=0.3, operation="chat")
        finally:
            middleware._client_gone.reset(token)

    with pytest.raises(HTTPException) as refused:
        asyncio.run(call())
    assert refused.value.status_code == 504
    assert stopped.wait(2)
