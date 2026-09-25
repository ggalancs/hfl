# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The llama-server backend and the per-engine dispatchers it needs.

A fake ``llama-server`` (a small Python HTTP server, pointed at with
``HFL_LLAMA_SERVER_BIN``) stands in for the real binary: it records its
command line, insists on the API key, and answers the endpoints the engine
uses. The real binary is exercised by hand (see the commit message).
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from hfl.engine.base import ChatMessage, GenerationConfig

FAKE = r"""
import json, os, socketserver, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
key = os.environ.get("LLAMA_API_KEY", "")
with open(os.environ["FAKE_ARGV_OUT"], "w") as out:
    json.dump({"argv": args, "key_in_env": bool(key), "pid": os.getpid()}, out)

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok"})
        if self.path == "/props" and self.headers.get("Authorization") == f"Bearer {key}":
            return self._send(200, {"default_generation_settings": {"n_ctx": 32768}})
        self._send(404, {})
    def do_POST(self):
        if self.headers.get("Authorization") != f"Bearer {key}":
            return self._send(401, {"error": "bad key"})
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        usage = {"prompt_tokens": 7, "completion_tokens": 2}
        timings = {"prompt_ms": 10.0, "predicted_ms": 20.0}
        if self.path == "/v1/chat/completions":
            if body.get("stream"):
                chunks = b"".join(
                    b"data: " + json.dumps({"choices": [{"delta": {"content": t}}]}).encode()
                    + b"\n\n" for t in ("Hel", "lo")
                )
                if (body.get("stream_options") or {}).get("include_usage"):
                    chunks += b"data: " + json.dumps({"choices": [], "usage": usage}).encode()
                    chunks += b"\n\n"
                chunks += b"data: [DONE]\n\n"
                return self._send(200, chunks, "text/event-stream")
            if body.get("tools"):
                call = {"id": "c1", "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
                message = {"role": "assistant", "content": "", "tool_calls": [call]}
            else:
                said = body["messages"][-1]["content"]
                message = {"role": "assistant", "content": "Hello " + said}
            return self._send(200, {"choices": [{"message": message, "finish_reason": "stop"}],
                                    "usage": usage, "timings": timings})
        if self.path == "/completion":
            if body.get("stream"):
                chunks = b"".join(
                    b"data: " + json.dumps({"content": t}).encode() + b"\n\n" for t in ("a", "b")
                ) + b"data: " + json.dumps(
                    {"content": "", "stop": True, "tokens_predicted": 2, "tokens_evaluated": 3}
                ).encode() + b"\n\n"
                return self._send(200, chunks, "text/event-stream")
            return self._send(200, {"content": "ab", "tokens_predicted": 2, "tokens_evaluated": 3,
                                    "timings": timings})
        self._send(404, {})

class Server(ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer.server_bind reverse-resolves the host (socket.getfqdn),
        # which can take tens of seconds on a machine with slow DNS.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "localhost", self.server_address[1]

Server(("127.0.0.1", port), H).serve_forever()
"""


@pytest.fixture
def fake_server(tmp_path, monkeypatch, temp_config):
    script = tmp_path / "llama-server"
    script.write_text(f"#!{sys.executable}\n{FAKE}")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    argv_out = tmp_path / "argv.json"
    monkeypatch.setenv("HFL_LLAMA_SERVER_BIN", str(script))
    monkeypatch.setenv("FAKE_ARGV_OUT", str(argv_out))
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(
        "hfl.engine.llama_cpp.resolve_n_ctx", lambda path, info, n, explicit: n or 8192
    )
    monkeypatch.setattr("hfl.engine.llama_cpp._read_gguf_model_info", lambda path: {})
    return model, argv_out


@pytest.fixture
def engine(fake_server):
    from hfl.engine.llama_server import LlamaServerEngine

    model, _ = fake_server
    eng = LlamaServerEngine()
    eng.load(str(model))
    yield eng
    eng.unload()


class TestProcess:
    def test_command_line_and_key(self, engine, fake_server):
        _, argv_out = fake_server
        seen = json.loads(argv_out.read_text())
        argv = seen["argv"]
        assert argv[argv.index("-c") + 1] == "8192"
        assert argv[argv.index("-np") + 1] == "4"
        assert argv[argv.index("--host") + 1] == "127.0.0.1"
        for flag in ("--kv-unified", "--jinja", "--no-webui", "--no-slots"):
            assert flag in argv
        assert argv[argv.index("--reasoning-format") + 1] == "none"
        # The key reaches the process through its environment, never argv.
        assert seen["key_in_env"] and "--api-key" not in argv
        assert engine.context_size == 8192 and engine.parallel_slots == 4
        assert engine.supports_concurrent_inference and engine.is_loaded

    def test_an_unreadable_header_leaves_the_context_to_llama_server(
        self, fake_server, monkeypatch
    ):
        """Without the ``gguf`` package HFL cannot read the model's limit, and
        its machine-sized default (262144 here) exceeded Phi-3.5's 131072.
        llama-server reads the GGUF itself; HFL asks it what it chose."""
        from hfl.engine.llama_server import LlamaServerEngine

        model, argv_out = fake_server
        monkeypatch.setattr("hfl.engine.llama_cpp._read_gguf_model_info", lambda path: None)
        eng = LlamaServerEngine()
        eng.load(str(model))
        try:
            assert "-c" not in json.loads(argv_out.read_text())["argv"]
            assert eng.context_size == 32768
        finally:
            eng.unload()

    def test_an_explicit_context_is_passed_even_then(self, fake_server, monkeypatch):
        from hfl.engine.llama_server import LlamaServerEngine

        model, argv_out = fake_server
        monkeypatch.setattr("hfl.engine.llama_cpp._read_gguf_model_info", lambda path: None)
        eng = LlamaServerEngine()
        eng.load(str(model), n_ctx=4096)
        try:
            argv = json.loads(argv_out.read_text())["argv"]
            assert argv[argv.index("-c") + 1] == "4096"
        finally:
            eng.unload()

    def test_the_process_answers_only_with_the_key(self, engine):
        import httpx

        base = str(engine._http().base_url)
        assert httpx.post(f"{base}/completion", json={"prompt": "x"}).status_code == 401

    def test_unload_stops_the_process(self, engine):
        proc = engine._proc
        engine.unload()
        assert proc.poll() is not None and not engine.is_loaded

    def test_hfl_num_parallel_sets_the_slots(self, fake_server, monkeypatch, temp_config):
        from hfl.engine.llama_server import LlamaServerEngine

        monkeypatch.setattr(temp_config, "queue_max_inflight", 8)
        import hfl.config

        monkeypatch.setattr(hfl.config.config, "queue_max_inflight", 8)
        eng = LlamaServerEngine()
        eng.load(str(fake_server[0]))
        try:
            assert eng.parallel_slots == 8
        finally:
            eng.unload()

    def test_a_missing_binary_is_a_clear_error(self, monkeypatch, tmp_path):
        from hfl.engine.llama_server import LlamaServerEngine

        monkeypatch.setenv("HFL_LLAMA_SERVER_BIN", str(tmp_path / "nope"))
        with pytest.raises(RuntimeError, match="llama-server was not found"):
            LlamaServerEngine().load(str(tmp_path / "m.gguf"))

    def test_a_killed_hfl_does_not_leave_llama_server_behind(self, fake_server, tmp_path):
        """HFL killed with SIGKILL cannot unload anything; the guard between
        it and llama-server must stop the child on its own."""
        import signal
        import subprocess

        model, argv_out = fake_server
        owner = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys, time\n"
                "from hfl.engine.llama_server import LlamaServerEngine\n"
                f"LlamaServerEngine().load({str(model)!r})\n"
                "print('ready', flush=True)\n"
                "time.sleep(600)\n",
            ],
            stdout=subprocess.PIPE,
            text=True,
            env={**os.environ, "HFL_HOME": str(tmp_path / "home")},
        )
        try:
            assert owner.stdout.readline().strip() == "ready"
            server_pid = json.loads(argv_out.read_text())["pid"]
            os.kill(server_pid, 0)  # alive while HFL is
            owner.send_signal(signal.SIGKILL)
            owner.wait(timeout=10)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    os.kill(server_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.2)
            else:
                os.kill(server_pid, signal.SIGKILL)  # this test's own mess
                pytest.fail("llama-server outlived the HFL process that started it")
        finally:
            if owner.poll() is None:
                owner.kill()


class TestRequests:
    def test_chat(self, engine):
        result = engine.chat([ChatMessage(role="user", content="you")])
        assert result.text == "Hello you"
        assert (result.tokens_prompt, result.tokens_generated) == (7, 2)
        assert result.eval_duration == 20_000_000 and result.prompt_eval_duration == 10_000_000

    def test_chat_stream(self, engine):
        stream = engine.chat_stream([ChatMessage(role="user", content="x")])
        assert list(stream) == ["Hel", "lo"]
        assert (stream.prompt_tokens, stream.completion_tokens) == (7, 2)

    def test_generate_stream_counts(self, engine):
        stream = engine.generate_stream("x")
        assert list(stream) == ["a", "b"]
        assert (stream.prompt_tokens, stream.completion_tokens) == (3, 2)

    def test_structured_tool_calls(self, engine):
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        result = engine.chat([ChatMessage(role="user", content="x")], tools=tools)
        assert result.tool_calls == [
            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        ]

    def test_streamed_tool_calls_reach_the_parser(self, engine):
        from hfl.api.tool_parsers import dispatch

        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        text = "".join(engine.chat_stream([ChatMessage(role="user", content="x")], tools=tools))
        # Whatever family the model's name suggests, the marker is read.
        for name in ("llama-3.1-8b", "mistral-7b", "phi-3.5-mini"):
            _, calls = dispatch(text, name, tools)
            assert calls == [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]

    def test_generate(self, engine):
        result = engine.generate("x", GenerationConfig(max_tokens=5))
        assert result.text == "ab" and result.tokens_generated == 2
        assert list(engine.generate_stream("x")) == ["a", "b"]

    def test_images_are_refused_plainly(self, engine):
        with pytest.raises(ValueError, match="images need the llama-cpp backend"):
            engine.chat([ChatMessage(role="user", content="x", images=[b"\x89PNG"])])


class TestSelection:
    def test_opt_in_for_gguf_only(self, monkeypatch, tmp_path):
        from hfl.engine.llama_server import LlamaServerEngine
        from hfl.engine.selector import select_engine

        monkeypatch.setenv("HFL_LLM_LIBRARY", "llama-server")
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"GGUF" + b"\0" * 64)
        assert isinstance(select_engine(gguf), LlamaServerEngine)
        folder = tmp_path / "hf"
        folder.mkdir()
        (folder / "config.json").write_text("{}")
        (folder / "model.safetensors").write_bytes(b"\0" * 16)
        assert not isinstance(select_engine(folder), LlamaServerEngine)


class TestFallback:
    """Without llama-cpp-python, a GGUF goes to llama-server if it is there."""

    @pytest.fixture
    def gguf(self, tmp_path):
        path = tmp_path / "m.gguf"
        path.write_bytes(b"GGUF" + b"\0" * 64)
        return path

    def _spec(self, monkeypatch, present: bool):
        import importlib.util

        if not present:
            # Hidden, not deleted: monkeypatch puts the same module object
            # back afterwards (a native extension must never be re-imported).
            monkeypatch.delitem(sys.modules, "llama_cpp", raising=False)
        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a: (
                real(name, *a) if name != "llama_cpp" else (object() if present else None)
            ),
        )

    def test_no_binding_but_a_server(self, gguf, monkeypatch, tmp_path):
        from hfl.engine.llama_server import LlamaServerEngine
        from hfl.engine.selector import select_engine

        monkeypatch.delenv("HFL_LLM_LIBRARY", raising=False)
        self._spec(monkeypatch, present=False)
        fake = tmp_path / "llama-server"
        fake.write_text("")
        monkeypatch.setenv("HFL_LLAMA_SERVER_BIN", str(fake))
        assert isinstance(select_engine(gguf), LlamaServerEngine)

    def test_the_binding_wins_when_installed(self, gguf, monkeypatch, tmp_path):
        from hfl.engine.llama_server import LlamaServerEngine
        from hfl.engine.selector import select_engine

        monkeypatch.delenv("HFL_LLM_LIBRARY", raising=False)
        self._spec(monkeypatch, present=True)
        fake = tmp_path / "llama-server"
        fake.write_text("")
        monkeypatch.setenv("HFL_LLAMA_SERVER_BIN", str(fake))
        assert not isinstance(select_engine(gguf), LlamaServerEngine)

    def test_neither_keeps_the_old_error_path(self, gguf, monkeypatch, tmp_path):
        from hfl.engine.llama_server import LlamaServerEngine
        from hfl.engine.selector import select_engine

        monkeypatch.delenv("HFL_LLM_LIBRARY", raising=False)
        self._spec(monkeypatch, present=False)
        monkeypatch.setenv("HFL_LLAMA_SERVER_BIN", str(tmp_path / "absent"))
        assert not isinstance(select_engine(gguf), LlamaServerEngine)


class _Engine:
    """Records how many of its calls overlap."""

    def __init__(self, concurrent: bool, slots: int = 0):
        self.supports_concurrent_inference = concurrent
        self.parallel_slots = slots
        self.active = self.peak = 0
        self.lock = threading.Lock()

    def work(self, seconds: float) -> str:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(seconds)
        with self.lock:
            self.active -= 1
        return "ok"


class TestDispatchers:
    @pytest.fixture(autouse=True)
    def fresh(self, temp_config):
        from hfl.core import reset_container

        reset_container()
        yield
        reset_container()

    def test_routing(self):
        from hfl.core import dispatcher_for, get_dispatcher

        serial, parallel = _Engine(False), _Engine(True, slots=3)
        assert dispatcher_for(serial) is get_dispatcher()
        assert dispatcher_for(None) is get_dispatcher()
        own = dispatcher_for(parallel)
        assert own is not get_dispatcher() and own is dispatcher_for(parallel)
        assert own.snapshot().max_inflight == 3

    @pytest.mark.parametrize(("concurrent", "peak"), [(True, 4), (False, 1)])
    def test_requests_overlap_only_on_a_concurrent_engine(self, concurrent, peak):
        from hfl.api.helpers import run_dispatched

        eng = _Engine(concurrent, slots=4)

        async def four() -> None:
            await asyncio.gather(*(run_dispatched(eng.work, 0.2) for _ in range(4)))

        asyncio.run(four())
        assert eng.peak == peak

    def test_a_concurrent_model_does_not_wait_behind_a_serial_one(self):
        from hfl.api.helpers import run_dispatched

        serial, parallel = _Engine(False), _Engine(True, slots=2)

        async def both() -> float:
            slow = asyncio.ensure_future(run_dispatched(serial.work, 1.0))
            await asyncio.sleep(0.05)
            started = time.monotonic()
            await run_dispatched(parallel.work, 0.05)
            waited = time.monotonic() - started
            await slow
            return waited

        assert asyncio.run(both()) < 0.5

    def test_unloading_drains_the_engines_own_queue(self):
        """_retire must wait for requests in the engine's own dispatcher —
        draining only the global one would unload a model mid-request."""
        from unittest.mock import MagicMock

        from hfl.api.state import ResidentModel, ServerState
        from hfl.core import dispatcher_for

        eng = _Engine(True, slots=2)
        eng.is_loaded = True
        eng.unloaded = False

        def unload() -> None:
            eng.unloaded = True

        eng.unload = unload
        state = ServerState()
        resident = ResidentModel(name="m", engine=eng, manifest=MagicMock(), footprint=0)

        async def scenario() -> tuple[bool, bool]:
            slot = dispatcher_for(eng).slot()
            await slot.__aenter__()  # a request is in flight on the engine's queue
            retire = asyncio.ensure_future(state._retire(resident, "test"))
            await asyncio.sleep(0.2)
            unloaded_mid_request = eng.unloaded
            await slot.__aexit__(None, None, None)
            await asyncio.wait_for(retire, 5)
            return unloaded_mid_request, eng.unloaded

        assert asyncio.run(scenario()) == (False, True)


def test_several_calls_written_back_as_markers_all_reach_the_client():
    """llama-server's structured calls come back as ``<tool_call>`` markers;
    for a family whose own parser does not read them (Llama, Mistral...), the
    loose JSON fallback would keep only the first."""
    from hfl.api.tool_parsers import dispatch

    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    text = (
        '<tool_call>{"name": "f", "arguments": {"i": 1}}</tool_call>'
        '<tool_call>{"name": "f", "arguments": {"i": 2}}</tool_call>'
    )
    for name in ("llama-3.1-8b", "mistral-7b"):
        _, calls = dispatch(text, name, tools)
        assert [c["function"]["arguments"]["i"] for c in calls] == [1, 2]


def test_os_environ_is_not_leaked_between_tests():
    assert "FAKE_ARGV_OUT" not in os.environ or Path(os.environ["FAKE_ARGV_OUT"]).exists()


def test_only_a_real_true_gets_its_own_dispatcher(temp_config):
    from unittest.mock import MagicMock

    from hfl.core import dispatcher_for, get_dispatcher, reset_container

    reset_container()
    try:
        assert dispatcher_for(MagicMock()) is get_dispatcher()  # truthy mock attribute
    finally:
        reset_container()
