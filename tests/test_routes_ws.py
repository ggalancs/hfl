# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Integration tests for ``WS /ws/chat`` (V4 F7).

The full duplex shape is covered with FastAPI's TestClient
WebSocket protocol: send frames, drain the responses, assert the
grammar.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


@pytest.fixture
def llm_manifest():
    from hfl.models.manifest import ModelManifest

    return ModelManifest(
        name="qwen-coder-7b",
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        local_path="/tmp/qwen-coder-7b.gguf",
        format="gguf",
        architecture="qwen",
        parameters="7B",
    )


def _wire_engine(manifest, *, tokens):
    state = get_state()
    engine = MagicMock()
    engine.is_loaded = True
    engine.chat_stream = MagicMock(return_value=iter(list(tokens)))
    state.engine = engine
    state.current_model = manifest
    return engine


def _drain_until(ws, predicate, *, max_frames: int = 50) -> list[dict]:
    """Read frames until ``predicate(frame)`` returns True.

    Cap on total frames so a misbehaving server doesn't hang the
    test forever.
    """
    out: list[dict] = []
    for _ in range(max_frames):
        raw = ws.receive_text()
        frame = json.loads(raw)
        out.append(frame)
        if predicate(frame):
            return out
    raise AssertionError(f"predicate never satisfied; last frames={out}")


class TestWsChatHappyPath:
    def test_sends_ready_then_tokens_then_done(self, client, llm_manifest):
        _wire_engine(llm_manifest, tokens=["Hel", "lo"])

        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "chat",
                        "model": llm_manifest.name,
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                )
            )
            frames = _drain_until(ws, lambda f: f["type"] == "done")

        types = [f["type"] for f in frames]
        assert types[0] == "ready"
        assert types[-1] == "done"
        # Two token frames, one per produced chunk.
        token_frames = [f for f in frames if f["type"] == "token"]
        assert [f["delta"] for f in token_frames] == ["Hel", "lo"]
        # Done frame counts the tokens.
        assert frames[-1]["tokens"] == 2

    def test_ping_pong(self, client, llm_manifest):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(json.dumps({"type": "ping"}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "pong"


class TestWsChatCancellation:
    def test_cancel_frame_interrupts_generation(self, client, llm_manifest):
        """Build a slow-token producer; send cancel mid-flight; ensure
        the server emits ``cancelled`` and stays open for the next
        chat."""

        def slow_tokens():
            # Yield 10 tokens with sleeps long enough that the cancel
            # frame can race in.
            for i in range(10):
                time.sleep(0.05)
                yield f"t{i}"

        state = get_state()
        engine = MagicMock(is_loaded=True)
        engine.chat_stream = MagicMock(return_value=slow_tokens())
        state.engine = engine
        state.current_model = llm_manifest

        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "chat",
                        "model": llm_manifest.name,
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                )
            )
            # Wait for ready, then cancel after one token.
            assert json.loads(ws.receive_text())["type"] == "ready"
            json.loads(ws.receive_text())  # consume one token
            ws.send_text(json.dumps({"type": "cancel"}))

            # Drain until we see cancelled or done.
            seen = []
            for _ in range(50):
                f = json.loads(ws.receive_text())
                seen.append(f)
                if f["type"] in ("cancelled", "done"):
                    break

            assert seen[-1]["type"] in ("cancelled", "done")
            # If the cancel raced before the producer finished we get
            # ``cancelled``; if the producer happened to finish first
            # we get ``done``. Either is valid; what matters is the
            # connection stays open.

            # Send a ping to confirm the connection is alive.
            ws.send_text(json.dumps({"type": "ping"}))
            assert json.loads(ws.receive_text())["type"] == "pong"

    def test_cancel_is_processed_mid_turn(self, client, llm_manifest):
        """CON-2: a cancel frame sent during generation is read by the receive
        loop and interrupts the turn deterministically — previously the loop
        was blocked awaiting the turn, so cancel was only seen after the engine
        had already finished (the turn always ended ``done``)."""

        def slow_tokens():
            for i in range(20):
                time.sleep(0.05)
                yield f"t{i}"

        state = get_state()
        engine = MagicMock(is_loaded=True)
        engine.chat_stream = MagicMock(return_value=slow_tokens())
        state.engine = engine
        state.current_model = llm_manifest

        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "chat",
                        "model": llm_manifest.name,
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                )
            )
            assert json.loads(ws.receive_text())["type"] == "ready"
            json.loads(ws.receive_text())  # one token in
            ws.send_text(json.dumps({"type": "cancel"}))
            seen = _drain_until(ws, lambda f: f["type"] in ("cancelled", "done"))

        # The turn ends as ``cancelled`` (not ``done``) and well before all
        # 20 tokens were produced — i.e. the cancel was honoured mid-flight.
        assert seen[-1]["type"] == "cancelled"
        assert sum(1 for f in seen if f["type"] == "token") < 20

    @pytest.mark.asyncio
    async def test_unexpected_turn_error_notifies_client(self, monkeypatch):
        """CON-2 follow-up: a background turn that dies on an *unexpected*
        post-``ready`` exception must still send the client a terminal
        ``error`` frame (not leave it hanging) via the done-callback."""
        import asyncio

        from hfl.api import routes_ws

        sent: list[dict] = []

        async def _fake_send(ws, frame):
            sent.append(frame)

        monkeypatch.setattr(routes_ws, "_send", _fake_send)

        async def _boom():
            raise RuntimeError("kaboom")

        task = asyncio.get_running_loop().create_task(_boom())
        await asyncio.sleep(0.01)
        assert task.done()

        routes_ws._log_drive_result(object(), task)  # ws unused by the fake send
        await asyncio.sleep(0.01)  # let the scheduled error-frame send run

        assert any(f["type"] == "error" for f in sent)


class TestWsChatThreadSafety:
    """CON regression: the worker-thread producer must marshal every queue
    enqueue back onto the loop (``asyncio.Queue`` is not thread-safe). A fast,
    bursty engine that produces tokens before the consumer re-registers its
    getter would otherwise race the queue internals and drop wakeups."""

    @pytest.mark.asyncio
    async def test_producer_enqueues_only_from_loop_thread(self, monkeypatch):
        import asyncio
        import threading

        from hfl.api import model_loader, routes_ws

        loop_thread_id = threading.get_ident()
        offending: list[int] = []
        real_put_nowait = asyncio.Queue.put_nowait

        def _tracking_put_nowait(self, item):
            if threading.get_ident() != loop_thread_id:
                offending.append(threading.get_ident())
            return real_put_nowait(self, item)

        monkeypatch.setattr(asyncio.Queue, "put_nowait", _tracking_put_nowait)

        class _FastEngine:
            """Bursty: 200 tokens, no sleeps — maximises the race window."""

            def chat_stream(self, msgs, cfg):
                for i in range(200):
                    yield f"t{i}"

        async def _fake_load_llm(model_name):
            return _FastEngine(), None

        monkeypatch.setattr(model_loader, "load_llm", _fake_load_llm)

        sent: list[dict] = []

        async def _fake_send(ws, frame):
            sent.append(frame)

        monkeypatch.setattr(routes_ws, "_send", _fake_send)

        cancel_event = asyncio.Event()
        frame = {
            "type": "chat",
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }
        await asyncio.wait_for(routes_ws._drive_chat(object(), frame, cancel_event), timeout=5.0)

        # Every token plus a terminal done must be delivered.
        token_frames = [f for f in sent if f.get("type") == "token"]
        assert len(token_frames) == 200
        assert any(f.get("type") == "done" for f in sent)
        # And no enqueue may have happened off the loop thread.
        assert offending == [], f"put_nowait called from worker thread(s): {offending}"


class TestWsChatValidation:
    def test_invalid_json_yields_error_frame(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text("not-json")
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "error"
            assert "JSON" in frame["message"]

    def test_unknown_frame_type_yields_error(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(json.dumps({"type": "fly-to-mars"}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "error"
            assert "unknown" in frame["message"]

    def test_chat_without_model_yields_error(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(
                json.dumps({"type": "chat", "messages": [{"role": "user", "content": "x"}]})
            )
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "error"
            assert "model" in frame["message"]

    def test_chat_with_empty_messages_yields_error(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(json.dumps({"type": "chat", "model": "x", "messages": []}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "error"
            assert "messages" in frame["message"]


class TestWsChatAuthAndOrigin:
    """V5 α1 + α2 — auth + origin gating on the WebSocket upgrade.

    The HTTP middleware doesn't see WebSocket frames, so the route
    enforces both policies inline before accepting frames.
    """

    def test_ws_rejects_when_api_key_required_but_missing(self, client):
        from hfl.api.state import get_state

        state = get_state()
        state.api_key = "secret-token-1234567890"
        try:
            with client.websocket_connect("/ws/chat") as ws:
                # Server accepts then closes with the reason in an
                # ``error`` frame.
                frame = json.loads(ws.receive_text())
                assert frame["type"] == "error"
                assert "unauthorized" in frame["message"]
        finally:
            state.api_key = None

    def test_ws_accepts_with_correct_api_key_in_query(self, client, llm_manifest):
        from hfl.api.state import get_state

        _wire_engine(llm_manifest, tokens=["a"])
        state = get_state()
        state.api_key = "right-token-1234567890"
        try:
            with client.websocket_connect("/ws/chat?api_key=right-token-1234567890") as ws:
                ws.send_text(json.dumps({"type": "ping"}))
                frame = json.loads(ws.receive_text())
                assert frame["type"] == "pong"
        finally:
            state.api_key = None

    def test_ws_accepts_with_authorization_bearer(self, client):
        from hfl.api.state import get_state

        state = get_state()
        state.api_key = "bearer-token-1234567890"
        try:
            with client.websocket_connect(
                "/ws/chat",
                headers={"authorization": "Bearer bearer-token-1234567890"},
            ) as ws:
                ws.send_text(json.dumps({"type": "ping"}))
                assert json.loads(ws.receive_text())["type"] == "pong"
        finally:
            state.api_key = None

    def test_ws_accepts_with_x_api_key_header(self, client):
        from hfl.api.state import get_state

        state = get_state()
        state.api_key = "x-key-token-1234567890"
        try:
            with client.websocket_connect(
                "/ws/chat",
                headers={"x-api-key": "x-key-token-1234567890"},
            ) as ws:
                ws.send_text(json.dumps({"type": "ping"}))
                assert json.loads(ws.receive_text())["type"] == "pong"
        finally:
            state.api_key = None

    def test_ws_rejects_wrong_origin(self, client, monkeypatch):
        """When ``cors_origins`` is set, an Origin header outside the
        allow-list closes the socket with policy violation."""
        from hfl.config import config

        monkeypatch.setattr(config, "cors_origins", ["https://app.example.com"])
        monkeypatch.setattr(config, "cors_allow_all", False)

        with client.websocket_connect(
            "/ws/chat",
            headers={"origin": "https://evil.example.com"},
        ) as ws:
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "error"
            assert "origin not allowed" in frame["message"]

    def test_ws_accepts_listed_origin(self, client, monkeypatch, llm_manifest):
        from hfl.config import config

        _wire_engine(llm_manifest, tokens=["a"])
        monkeypatch.setattr(config, "cors_origins", ["https://app.example.com"])
        monkeypatch.setattr(config, "cors_allow_all", False)

        with client.websocket_connect(
            "/ws/chat",
            headers={"origin": "https://app.example.com"},
        ) as ws:
            ws.send_text(json.dumps({"type": "ping"}))
            assert json.loads(ws.receive_text())["type"] == "pong"

    def test_ws_wildcard_origin_accepts_anything(self, client, monkeypatch, llm_manifest):
        from hfl.config import config

        _wire_engine(llm_manifest, tokens=["a"])
        monkeypatch.setattr(config, "cors_allow_all", True)
        monkeypatch.setattr(config, "cors_origins", ["*"])

        with client.websocket_connect(
            "/ws/chat",
            headers={"origin": "https://anything.example.com"},
        ) as ws:
            ws.send_text(json.dumps({"type": "ping"}))
            assert json.loads(ws.receive_text())["type"] == "pong"

    def test_ws_no_origin_header_passes_when_no_api_key(self, client, llm_manifest):
        """Same-origin connections (no Origin header) are accepted by
        default — that's how a CLI client or curl websocat looks."""
        _wire_engine(llm_manifest, tokens=["a"])
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(json.dumps({"type": "ping"}))
            assert json.loads(ws.receive_text())["type"] == "pong"


class TestWsChatCancelTelemetry:
    """V6 ν2 — every cancel bumps a counter; the orphans subset is
    flagged when the engine thread was still running at cancel time.

    Note: a TestClient WS cancel runs in-process under the GIL so
    the racing between the cancel event and the producer thread is
    nondeterministic. We split the contract into deterministic
    pieces:

    1. ``record_ws_cancel`` itself increments the counters as
       documented.
    2. The Prometheus export reflects them.
    3. The handler invokes ``record_ws_cancel`` *whenever* the
       cancellation path runs (smoke-tested elsewhere).
    """

    def test_record_ws_cancel_increments_total(self):
        from hfl.metrics import get_metrics, reset_metrics

        reset_metrics()
        m = get_metrics()
        m.record_ws_cancel(orphaned_slot=False)
        assert m.ws_cancels_total == 1
        assert m.ws_cancel_orphans_total == 0

    def test_record_ws_cancel_orphan_increments_both(self):
        from hfl.metrics import get_metrics, reset_metrics

        reset_metrics()
        m = get_metrics()
        m.record_ws_cancel(orphaned_slot=True)
        assert m.ws_cancels_total == 1
        assert m.ws_cancel_orphans_total == 1

    def test_prometheus_exports_ws_cancel_metrics(self, client):
        from hfl.metrics import get_metrics, reset_metrics

        reset_metrics()
        m = get_metrics()
        m.record_ws_cancel(orphaned_slot=True)
        m.record_ws_cancel(orphaned_slot=False)

        body = client.get("/metrics").text
        assert "hfl_ws_cancels_total" in body
        assert "hfl_ws_cancel_orphans_total" in body
        assert "hfl_ws_cancels_total 2" in body
        assert "hfl_ws_cancel_orphans_total 1" in body


class TestWsChatPersistentConnection:
    def test_multiple_chats_in_one_connection(self, client, llm_manifest):
        _wire_engine(llm_manifest, tokens=["a"])

        with client.websocket_connect("/ws/chat") as ws:
            for _ in range(2):
                # Re-arm the iterator each turn.
                state = get_state()
                state.engine.chat_stream = MagicMock(return_value=iter(["a"]))

                ws.send_text(
                    json.dumps(
                        {
                            "type": "chat",
                            "model": llm_manifest.name,
                            "messages": [{"role": "user", "content": "hi"}],
                        }
                    )
                )
                frames = _drain_until(ws, lambda f: f["type"] == "done")
                assert frames[-1]["type"] == "done"
