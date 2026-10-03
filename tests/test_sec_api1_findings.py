# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Regression tests for the API security findings fixed in group "api1".

Each class reproduces one finding against the real app. ``TestClient``
presents ``client.host == "testclient"`` — a REMOTE peer for the owner
boundary; ``client=("127.0.0.1", ...)`` makes it the owner.
"""

from __future__ import annotations

import contextlib
import json
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state
from hfl.engine.base import GenerationResult


@pytest.fixture
def remote(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


@pytest.fixture
def owner(temp_config):
    reset_state()
    yield TestClient(app, client=("127.0.0.1", 50000))
    reset_state()


def _manifest(name: str = "m"):
    from hfl.models.manifest import ModelManifest

    return ModelManifest(name=name, repo_id="org/m", local_path="/tmp/x.gguf", format="gguf")


def _install(engine, manifest=None):
    state = get_state()
    state.engine = engine
    state.current_model = manifest or _manifest()
    return engine


def _chat_frame(model: str = "m", **options):
    frame = {"type": "chat", "model": model, "messages": [{"role": "user", "content": "hi"}]}
    if options:
        frame["options"] = options
    return json.dumps(frame)


def _drain(ws, until, limit: int = 60) -> list[dict]:
    out: list[dict] = []
    for _ in range(limit):
        frame = json.loads(ws.receive_text())
        out.append(frame)
        if until(frame):
            return out
    raise AssertionError(f"never saw the expected frame: {out}")


# ---------------------------------------------------------------------------
# 1. /ws/chat: one generation per socket, through the dispatcher, bounded
# ---------------------------------------------------------------------------


class TestWsGenerationBounds:
    def test_new_turn_waits_for_the_cancelled_producer(self, remote):
        """chat → cancel → chat used to start a second engine call while the
        first (which cannot be preempted) was still running: one socket
        stacked unbounded generations in the thread pool."""
        gate = threading.Event()
        first_done = threading.Event()
        second_started_after_first: list[bool] = []

        class _Engine:
            is_loaded = True
            calls = 0

            def chat_stream(self, msgs, cfg):
                type(self).calls += 1
                if type(self).calls == 1:
                    yield "t0"
                    gate.wait(10)  # ignores cancel, like llama-cpp-python
                    first_done.set()
                    return
                second_started_after_first.append(first_done.is_set())
                yield "x"

        _install(_Engine())
        with remote.websocket_connect("/ws/chat") as ws:
            ws.send_text(_chat_frame())
            _drain(ws, lambda f: f["type"] == "token")
            ws.send_text(json.dumps({"type": "cancel"}))
            _drain(ws, lambda f: f["type"] == "cancelled")
            ws.send_text(_chat_frame())
            threading.Timer(0.4, gate.set).start()
            _drain(ws, lambda f: f["type"] == "done")
        assert second_started_after_first == [True]

    def test_turn_goes_through_the_dispatcher(self, remote, monkeypatch):
        """A full queue refuses the turn instead of generating outside it."""
        import hfl.core
        from hfl.engine.dispatcher import QueueFullError

        class _Full:
            @contextlib.asynccontextmanager
            async def slot(self):
                raise QueueFullError(depth=4, max_queued=4, retry_after=3)
                yield  # pragma: no cover

            def snapshot(self):  # suggest_parallel may ask
                return MagicMock(in_flight=0, max_inflight=1)

        monkeypatch.setattr(hfl.core, "dispatcher_for", lambda engine: _Full())
        engine = _install(MagicMock(is_loaded=True))
        engine.chat_stream = MagicMock(return_value=iter(["a"]))
        with remote.websocket_connect("/ws/chat") as ws:
            ws.send_text(_chat_frame())
            frames = _drain(ws, lambda f: f["type"] in ("error", "done"))
        assert frames[-1]["type"] == "error"
        assert frames[-1].get("code") == "queue_full"
        engine.chat_stream.assert_not_called()

    def test_default_max_tokens_is_bounded(self, remote):
        seen: list[int] = []

        def _stream(msgs, cfg):
            seen.append(cfg.max_tokens)
            yield "a"

        engine = _install(MagicMock(is_loaded=True))
        engine.chat_stream = MagicMock(side_effect=_stream)
        with remote.websocket_connect("/ws/chat") as ws:
            ws.send_text(_chat_frame())
            _drain(ws, lambda f: f["type"] == "done")
        assert seen and seen[0] > 0

    def test_explicit_max_tokens_is_kept(self, remote):
        seen: list[int] = []

        def _stream(msgs, cfg):
            seen.append(cfg.max_tokens)
            yield "a"

        engine = _install(MagicMock(is_loaded=True))
        engine.chat_stream = MagicMock(side_effect=_stream)
        with remote.websocket_connect("/ws/chat") as ws:
            ws.send_text(_chat_frame(max_tokens=7))
            _drain(ws, lambda f: f["type"] == "done")
        assert seen == [7]


# ---------------------------------------------------------------------------
# 2. Agent loop and MCP tool schemas are the owner's
# ---------------------------------------------------------------------------


class _Tool:
    def to_ollama_tool(self) -> dict:
        return {"type": "function", "function": {"name": "fs__secret", "parameters": {}}}


def _fake_mcp(monkeypatch):
    import hfl.mcp.client as mcp_client

    calls: list[str] = []

    async def _call(name, args):
        calls.append(name)
        return "ran"

    fake = MagicMock()
    fake.list_tools = MagicMock(return_value=[_Tool()])
    fake.call_tool = _call
    monkeypatch.setattr(mcp_client, "get_client", lambda: fake)
    return calls


class TestAgentLoopOwnerOnly:
    def test_remote_peer_cannot_run_the_agent_loop(self, remote, monkeypatch):
        from hfl.config import config as cfg

        monkeypatch.setattr(cfg, "allow_agent_loop", True)
        calls = _fake_mcp(monkeypatch)
        engine = _install(MagicMock(is_loaded=True))
        engine.chat = MagicMock(
            side_effect=[
                GenerationResult(
                    text="",
                    tokens_generated=1,
                    tool_calls=[{"id": "c1", "function": {"name": "fs__secret", "arguments": {}}}],
                ),
                GenerationResult(text="done", tokens_generated=1),
            ]
        )
        resp = remote.post(
            "/api/chat",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "agent_loop": True,
            },
        )
        assert resp.status_code == 403, resp.text
        assert calls == []

    def test_owner_passes_the_gate(self, owner, monkeypatch):
        from hfl.config import config as cfg

        monkeypatch.setattr(cfg, "allow_agent_loop", True)
        calls = _fake_mcp(monkeypatch)
        engine = _install(MagicMock(is_loaded=True))
        engine.chat = MagicMock(
            side_effect=[
                GenerationResult(
                    text="",
                    tokens_generated=1,
                    tool_calls=[{"id": "c1", "function": {"name": "fs__secret", "arguments": {}}}],
                ),
                GenerationResult(text="done", tokens_generated=1),
            ]
        )
        resp = owner.post(
            "/api/chat",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "agent_loop": True,
            },
        )
        assert resp.status_code == 200, resp.text
        assert calls == ["fs__secret"]

    @pytest.mark.parametrize("who,expected", [("remote", None), ("owner", ["fs__secret"])])
    def test_mcp_schemas_folded_only_for_the_owner(self, who, expected, request, monkeypatch):
        client = request.getfixturevalue(who)
        _fake_mcp(monkeypatch)
        engine = _install(MagicMock(is_loaded=True))
        engine.chat = MagicMock(return_value=GenerationResult(text="ok", tokens_generated=1))
        resp = client.post(
            "/api/chat",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        assert resp.status_code == 200, resp.text
        tools = engine.chat.call_args.kwargs.get("tools")
        names = [t["function"]["name"] for t in tools] if tools else None
        assert names == expected


# ---------------------------------------------------------------------------
# 3. POST /api/blobs is an owner operation (also from a web page)
# ---------------------------------------------------------------------------


class TestBlobsOwnerOnly:
    DIGEST = "sha256:" + "0" * 64

    def test_remote_peer_refused(self, remote):
        resp = remote.post(f"/api/blobs/{self.DIGEST}", content=b"x")
        assert resp.status_code == 403

    def test_web_page_on_loopback_refused(self, owner):
        resp = owner.post(
            f"/api/blobs/{self.DIGEST}",
            content=b"x",
            headers={"Origin": "https://evil.example", "Content-Type": "text/plain"},
        )
        assert resp.status_code == 403
        assert resp.json()["detail"]["code"] == "cross_origin_admin_forbidden"

    def test_owner_still_uploads(self, owner):
        import hashlib

        body = b"hello blob"
        digest = "sha256:" + hashlib.sha256(body).hexdigest()
        assert owner.post(f"/api/blobs/{digest}", content=body).status_code == 201


# ---------------------------------------------------------------------------
# 4. keep_alive that unloads or pins forever is the owner's
# ---------------------------------------------------------------------------


def _ok_engine():
    engine = MagicMock(is_loaded=True)
    engine.generate = MagicMock(return_value=GenerationResult(text="ok", tokens_generated=1))
    engine.chat = MagicMock(return_value=GenerationResult(text="ok", tokens_generated=1))
    return engine


class TestKeepAliveOwnerOnly:
    @pytest.mark.parametrize("route", ["/api/generate", "/api/chat"])
    @pytest.mark.parametrize("prompt", [True, False], ids=["with-prompt", "preload"])
    def test_remote_keep_alive_zero_does_not_unload(self, remote, route, prompt):
        _install(_ok_engine())
        state = get_state()
        state.evict = AsyncMock(return_value=True)  # type: ignore[method-assign]
        state.cleanup = AsyncMock()  # type: ignore[method-assign]
        body: dict = {"model": "m", "stream": False, "keep_alive": 0}
        if route == "/api/generate":
            body["prompt"] = "hi" if prompt else ""
        else:
            body["messages"] = [{"role": "user", "content": "hi"}] if prompt else []
        resp = remote.post(route, json=body)
        assert resp.status_code == 200, resp.text
        state.evict.assert_not_called()
        state.cleanup.assert_not_called()

    def test_remote_keep_alive_minus_one_does_not_pin(self, remote):
        _install(_ok_engine())
        state = get_state()
        state.set_keep_alive = MagicMock()  # type: ignore[method-assign]
        resp = remote.post(
            "/api/generate", json={"model": "m", "prompt": "hi", "stream": False, "keep_alive": -1}
        )
        assert resp.status_code == 200, resp.text
        assert all(c.args[1] is not None for c in state.set_keep_alive.call_args_list)

    def test_owner_keep_alive_zero_still_unloads(self, owner):
        _install(_ok_engine())
        state = get_state()
        state.evict = AsyncMock(return_value=True)  # type: ignore[method-assign]
        resp = owner.post(
            "/api/generate", json={"model": "m", "prompt": "hi", "stream": False, "keep_alive": 0}
        )
        assert resp.status_code == 200, resp.text
        state.evict.assert_awaited()

    def test_remote_positive_keep_alive_still_applies(self, remote):
        _install(_ok_engine())
        resp = remote.post(
            "/api/generate",
            json={"model": "m", "prompt": "hi", "stream": False, "keep_alive": "5m"},
        )
        assert resp.status_code == 200
        assert get_state().keep_alive_deadline_for("m") is not None


# ---------------------------------------------------------------------------
# 5. DNS rebinding: Host must be a loopback name on a loopback server
# ---------------------------------------------------------------------------


class TestHostValidation:
    @pytest.fixture
    def loopback(self, temp_config):
        # The server address TestClient puts in the ASGI scope comes from
        # base_url: this is a server listening on 127.0.0.1:11434.
        reset_state()
        yield TestClient(app, base_url="http://127.0.0.1:11434")
        reset_state()

    def test_rebound_host_is_refused(self, loopback):
        resp = loopback.get("/api/tags", headers={"Host": "rebind.evil.example:11434"})
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "host_not_allowed"

    @pytest.mark.parametrize(
        "host", ["127.0.0.1:11434", "localhost:11434", "[::1]:11434", "localhost", "LOCALHOST"]
    )
    def test_loopback_names_pass(self, loopback, host):
        assert loopback.get("/api/tags", headers={"Host": host}).status_code == 200

    def test_listed_origin_host_passes(self, loopback, monkeypatch):
        from hfl.config import config as cfg

        monkeypatch.setattr(cfg, "cors_origins", ["https://chat.example.org"])
        resp = loopback.get("/api/tags", headers={"Host": "chat.example.org"})
        assert resp.status_code == 200

    def test_suffix_of_a_loopback_name_is_refused(self, loopback):
        resp = loopback.get("/api/tags", headers={"Host": "localhost.evil.example"})
        assert resp.status_code == 403

    def test_websocket_with_rebound_host_is_refused(self, loopback):
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            # websocket_connect ignores base_url: the server address comes
            # from this absolute URL.
            with loopback.websocket_connect(
                "ws://127.0.0.1:11434/ws/chat", headers={"Host": "rebind.evil.example:11434"}
            ) as ws:
                ws.send_text(json.dumps({"type": "ping"}))
                ws.receive_text()

    def test_websocket_with_loopback_host_passes(self, loopback):
        with loopback.websocket_connect(
            "ws://127.0.0.1:11434/ws/chat", headers={"Host": "localhost:11434"}
        ) as ws:
            ws.send_text(json.dumps({"type": "ping"}))
            assert json.loads(ws.receive_text())["type"] == "pong"

    def test_recorded_loopback_bind_is_enforced(self, remote, monkeypatch):
        """``start_server`` records the bind; it decides over the scope."""
        import hfl.api.server as server

        monkeypatch.setattr(server, "_bound_host", "127.0.0.1", raising=False)
        assert remote.get("/api/tags", headers={"Host": "evil.example"}).status_code == 403
        assert remote.get("/api/tags", headers={"Host": "127.0.0.1:11434"}).status_code == 200

    def test_server_bound_to_all_interfaces_is_left_alone(self, loopback, monkeypatch):
        import hfl.api.server as server

        monkeypatch.setattr(server, "_bound_host", "0.0.0.0", raising=False)
        resp = loopback.get("/api/tags", headers={"Host": "models.lan:11434"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 6. /health/deep?probe=true: key, dispatcher, owner-only process figures
# ---------------------------------------------------------------------------


class TestHealthDeep:
    def test_probe_requires_the_key(self, remote):
        get_state().api_key = "s3cret"
        assert remote.get("/health/deep?probe=true").status_code == 401
        assert remote.get("/health/deep").status_code == 200
        ok = remote.get("/health/deep?probe=true", headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200

    def test_probe_runs_through_the_dispatcher(self, remote, monkeypatch):
        import hfl.core

        used: list[bool] = []
        real = hfl.core.dispatcher_for

        def _spy(engine):
            used.append(True)
            return real(engine)

        monkeypatch.setattr(hfl.core, "dispatcher_for", _spy)
        _install(_ok_engine())
        body = remote.get("/health/deep?probe=true").json()
        assert body["llm"]["probe"] == "ok"
        assert used

    def test_process_figures_only_for_the_owner(self, remote, owner):
        pytest.importorskip("psutil")
        assert "memory_mb" not in (remote.get("/health/deep").json().get("system") or {})
        assert "memory_mb" in owner.get("/health/deep").json()["system"]


# ---------------------------------------------------------------------------
# 7. A failed WebSocket key counts like a failed HTTP key
# ---------------------------------------------------------------------------


class TestWsAuthBackoff:
    def test_failed_ws_key_is_recorded(self, remote):
        import hfl.api.server as server

        server._AUTH_FAILURES.clear()
        get_state().api_key = "s3cret"
        with remote.websocket_connect("/ws/chat?api_key=wrong") as ws:
            assert json.loads(ws.receive_text())["type"] == "error"
        assert server._AUTH_FAILURES.get("testclient") == 1
        server._AUTH_FAILURES.clear()
