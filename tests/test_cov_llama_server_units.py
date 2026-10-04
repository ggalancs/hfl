# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl.engine.llama_server`` piece by piece, without a process.

``tests/test_llama_server_engine.py`` drives the engine through a fake
llama-server binary; this file covers what that end-to-end run does not
reach — the flag builders, the device probe, the error paths of the
prompt cache, the template probes and the restart-on-death plumbing —
with a stand-in HTTP client set on the engine, so each case is quick and
says exactly what it checks.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import types

import httpx
import pytest

import hfl.engine.llama_server as ls
from hfl.engine import cancel
from hfl.engine.base import ChatMessage, GenerationConfig

# ---------------------------------------------------------------- helpers


class TestFlagBuilders:
    def test_flash_attention(self, monkeypatch):
        monkeypatch.delenv("HFL_FLASH_ATTENTION", raising=False)
        monkeypatch.delenv("OLLAMA_FLASH_ATTENTION", raising=False)
        assert ls._flash_attention_args(None) == []
        assert ls._flash_attention_args(True) == ["-fa", "on"]
        assert ls._flash_attention_args(False) == ["-fa", "off"]
        monkeypatch.setenv("OLLAMA_FLASH_ATTENTION", "0")
        assert ls._flash_attention_args(None) == ["-fa", "off"]

    def test_kv_cache(self, temp_config, caplog):
        assert ls._kv_cache_args("q4_0") == ["-ctk", "q4_0", "-ctv", "q4_0"]
        assert ls._kv_cache_args("F16") == []
        with caplog.at_level(logging.WARNING, logger="hfl.engine.llama_server"):
            assert ls._kv_cache_args("iq2") == []
        assert "unsupported by llama-server" in caplog.text

    def test_sampling_and_response_format(self):
        body = ls._sampling(GenerationConfig(seed=7, stop=["\n\n"]))
        assert body["seed"] == 7 and body["stop"] == ["\n\n"]
        assert "seed" not in ls._sampling(GenerationConfig())
        assert ls._response_format("json") == {"type": "json_object"}
        assert ls._response_format({"type": "integer"}) == {
            "type": "json_schema",
            "json_schema": {"schema": {"type": "integer"}},
        }
        assert ls._response_format("GBNF:root ::= x") is None

    def test_argv_slots(self):
        assert ls._argv_slots(["x", "-np", "3"]) == 3
        assert ls._argv_slots(["x"]) == 0
        assert ls._argv_slots(["x", "-np"]) == 0

    def test_tool_calls_and_logprobs(self):
        calls = [{"function": {"name": "w", "arguments": "{broken"}}]
        assert ls._canonical_tool_calls(calls) == [{"function": {"name": "w", "arguments": {}}}]
        assert ls._logprob_entries(None, 1) is None

    def test_a_tool_result_without_a_name(self):
        wire = ls._wire_messages([ChatMessage("tool", "18C")])
        assert wire == [{"role": "tool", "content": "18C", "tool_call_id": ""}]


class TestProbes:
    @pytest.fixture(autouse=True)
    def _fresh(self):
        ls.has_gpu.cache_clear()
        ls._help_text.cache_clear()
        yield
        ls.has_gpu.cache_clear()
        ls._help_text.cache_clear()

    def _run(self, monkeypatch, stdout="", returncode=0, error=None):
        def run(argv, **kwargs):
            if error is not None:
                raise error
            return types.SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)

        monkeypatch.setattr(ls.subprocess, "run", run)

    def test_a_cpu_only_build(self, monkeypatch):
        self._run(monkeypatch, "Available devices:\n  BLAS: Accelerate\n  CPU: cores\n")
        assert ls.has_gpu("/bin/cpu-only") is False

    def test_a_gpu_build(self, monkeypatch):
        self._run(monkeypatch, "Available devices:\n  MTL0: Apple M3 Max (98304 MiB)\n")
        assert ls.has_gpu("/bin/metal") is True

    def test_unknown_counts_as_a_gpu(self, monkeypatch):
        self._run(monkeypatch, error=OSError("cannot run"))
        assert ls.has_gpu("/bin/broken") is True
        ls.has_gpu.cache_clear()
        self._run(monkeypatch, "usage: ...", returncode=1)
        assert ls.has_gpu("/bin/old") is True

    def test_help_text(self, monkeypatch):
        self._run(monkeypatch, "--spec-type TYPE")
        assert ls._help_text("/bin/a") == "--spec-type TYPE"
        self._run(monkeypatch, error=subprocess.TimeoutExpired("x", 30))
        assert ls._help_text("/bin/b") == ""


class TestStopServer:
    def test_a_process_ignoring_sigterm_is_killed_with_its_group(self, monkeypatch):
        class Proc:
            pid = 4242
            killed = False

            def poll(self):
                return None

            def send_signal(self, sig):
                self.signal = sig

            def wait(self, timeout):
                if not self.killed:
                    raise subprocess.TimeoutExpired("llama-server", timeout)

            def kill(self):
                self.killed = True

        def killpg(pid, sig):
            raise ProcessLookupError(pid)

        monkeypatch.setattr(ls.os, "killpg", killpg)
        monkeypatch.setattr(ls, "_STOP_WAIT", 0.01)
        proc = Proc()
        ls.stop_server(proc)
        # The group was gone already: the guard itself is killed.
        assert proc.killed


# ----------------------------------------------------------- stand-in HTTP


class _Resp:
    def __init__(self, body=None, status=200, lines=()):
        self._body, self.status_code, self._lines = body, status, list(lines)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://x")
            raise httpx.HTTPStatusError(
                "error", request=request, response=httpx.Response(self.status_code)
            )

    def iter_lines(self):
        return iter(self._lines)


class _Stream:
    def __init__(self, response):
        self.response = response
        self.exits: list = []

    def __enter__(self):
        return self.response

    def __exit__(self, *exc):
        self.exits.append(exc)
        return False


class _Client:
    """Routes ``path`` to a callable or a fixed response; records requests."""

    def __init__(self, routes=None):
        self.routes = routes or {}
        self.sent: list[tuple[str, str, dict]] = []
        self.closed = False

    def _answer(self, method, path, kwargs):
        self.sent.append((method, path, kwargs))
        route = self.routes.get(path.split("?")[0])
        if route is None:
            raise AssertionError(f"unexpected {method} {path}")
        out = route(kwargs) if callable(route) else route
        if isinstance(out, Exception):
            raise out
        return out

    def post(self, path, **kwargs):
        return self._answer("POST", path, kwargs)

    def get(self, path, **kwargs):
        return self._answer("GET", path, kwargs)

    def stream(self, method, path, **kwargs):
        return _Stream(self._answer(method, path, kwargs))

    def close(self):
        self.closed = True


def _engine(routes=None, **attrs) -> ls.LlamaServerEngine:
    engine = ls.LlamaServerEngine()
    engine._client = _Client(routes)
    engine._model_path = "/m/qwen.gguf"
    engine._argv = ["/bin/llama-server", "-m", "/m/qwen.gguf", "-np", "2"]
    for key, value in attrs.items():
        setattr(engine, key, value)
    return engine


def _sse(*events) -> list[str]:
    return [f"data: {json.dumps(e)}" for e in events] + ["data: [DONE]"]


class TestPromptCacheErrors:
    def test_a_slot_that_will_not_restore_is_dropped(self, tmp_path, caplog):
        (tmp_path / "slot-0.bin").write_bytes(b"KV")
        (tmp_path / "slot-1.bin").write_bytes(b"KV")
        routes = {
            "/slots/0": httpx.ReadTimeout("slow"),
            "/slots/1": _Resp({"n_restored": 5}),
        }
        engine = _engine(routes, _cache_dir=tmp_path)
        with caplog.at_level(logging.INFO, logger="hfl.engine.llama_server"):
            engine._restore_slots(3)
        assert not (tmp_path / "slot-0.bin").exists()
        assert (tmp_path / "slot-1.bin").exists()
        assert "slot 0 not restored" in caplog.text
        assert "5 tokens restored" in caplog.text

    def test_a_slot_that_will_not_save_is_not_kept(self, tmp_path, temp_config, caplog):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "slot-0.bin").write_bytes(b"old")
        routes = {"/slots/0": _Resp(ValueError("not json")), "/slots/1": _Resp({"n_saved": 0})}
        engine = _engine(routes, _cache_dir=cache)
        with caplog.at_level(logging.WARNING, logger="hfl.engine.llama_server"):
            engine._save_slots()
        assert "slot 0 not saved" in caplog.text
        assert list(cache.iterdir()) == []

    def test_nothing_to_save_without_a_cache(self):
        engine = _engine()
        engine._save_slots()
        assert engine._client.sent == []


class TestTemplateProbes:
    def test_props_unreachable(self):
        engine = _engine({"/props": httpx.ConnectTimeout("down")})
        assert engine._props() == {}
        engine = _engine({"/props": _Resp(["not", "a", "dict"])})
        assert engine._props() == {}

    def test_no_template_reported(self, tmp_path):
        engine = _engine({"/props": _Resp({"bos_token": "<s>"})})
        assert engine._template_with_bos(tmp_path, "m") is None

    def test_a_known_mistake_is_corrected_in_a_copy(self, tmp_path, caplog):
        from hfl.models.chat_template import _TEMPLATE_REPAIRS

        wrong, right = _TEMPLATE_REPAIRS[0]
        template = "{{ bos_token }}" + wrong
        engine = _engine({"/props": _Resp({"chat_template": template, "bos_token": "<s>"})})
        with caplog.at_level(logging.INFO, logger="hfl.engine.llama_server"):
            path = engine._template_with_bos(tmp_path, "coder")
        assert path == tmp_path / "coder.jinja"
        assert path.read_text(encoding="utf-8") == "{{ bos_token }}" + right
        assert "known mistake" in caplog.text

    def test_bos_unknown_when_tokenize_fails(self):
        engine = _engine({"/tokenize": _Resp({"no_tokens": []})})
        assert engine._adds_bos() is False

    def test_tools_probed_from_the_template_without_caps(self, monkeypatch):
        import hfl.engine.llama_cpp as lc

        seen = []
        monkeypatch.setattr(
            lc, "_template_renders_tools", lambda t, fmt: seen.append((t, fmt)) or True
        )
        engine = _engine({"/props": _Resp({"chat_template": "T", "n_ctx": 1024})})
        engine._template_knows_tools = False
        engine._read_template()
        assert engine._template_knows_tools is True and seen == [("T", None)]
        assert engine._reported_ctx() == 1024


class TestRevival:
    def test_requests_without_a_process(self):
        engine = ls.LlamaServerEngine()
        with pytest.raises(RuntimeError, match="not loaded"):
            engine._http()

    def test_died_is_false_without_a_process_or_while_running(self):
        engine = _engine()
        assert engine._died() is False

        class Running:
            def wait(self, timeout):
                raise subprocess.TimeoutExpired("llama-server", timeout)

        engine._proc = Running()
        assert engine._died() is False

    @pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
    def test_how_it_ended_names_the_signal(self):
        assert ls._how_it_ended(128 + signal.SIGKILL) == "was killed (SIGKILL; out of memory?)"
        assert ls._how_it_ended(128 + signal.SIGSEGV) == "was killed (SIGSEGV)"
        assert ls._how_it_ended(128 + 99) == "was killed (signal 99)"
        assert ls._how_it_ended(0) == "exited (code 0)"
        assert ls._how_it_ended(1) == "exited (code 1)"
        assert ls._how_it_ended(None) == "exited (code None)"

    def test_revive_only_the_process_that_died(self):
        engine = _engine()
        engine._proc = object()
        engine._revive(object())  # another one: already started again
        assert not engine._client.closed

    def test_a_refused_request_is_raised_when_the_process_lives(self, monkeypatch):
        engine = _engine({"/props": httpx.ConnectError("refused"), "/x": httpx.ConnectError("no")})
        monkeypatch.setattr(engine, "_died", lambda: False)
        with pytest.raises(httpx.ConnectError):
            engine._get("/props")
        with pytest.raises(httpx.ConnectError):
            engine._post("/x")
        with pytest.raises(httpx.ConnectError):
            with engine._stream("POST", "/x"):
                pass

    def test_sent_again_once_it_was_started_again(self, monkeypatch):
        answers = iter([httpx.RemoteProtocolError("reset"), _Resp({"ok": 1})])
        engine = _engine({"/props": lambda kw: next(answers)})
        monkeypatch.setattr(engine, "_died", lambda: True)
        assert engine._get("/props").json() == {"ok": 1}
        answers = iter([httpx.ReadError("reset"), _Resp(lines=["data: 1"])])
        engine._client.routes["/v1/chat/completions"] = lambda kw: next(answers)
        with engine._stream("POST", "/v1/chat/completions") as response:
            assert list(response.iter_lines()) == ["data: 1"]


# --------------------------------------------------------------- requests

USAGE = {"prompt_tokens": 7, "completion_tokens": 2}


class TestRequests:
    def test_chat_with_a_response_format_and_logprobs(self):
        reply = {
            "choices": [
                {
                    "message": {"content": "{}"},
                    "finish_reason": "length",
                    "logprobs": {"content": [{"token": "{}", "logprob": -0.5, "top_logprobs": []}]},
                }
            ],
            "usage": USAGE,
        }
        engine = _engine({"/v1/chat/completions": _Resp(reply)})
        result = engine.chat(
            [ChatMessage("user", "q")], GenerationConfig(response_format="json", logprobs=0)
        )
        body = engine._client.sent[-1][2]["json"]
        assert body["response_format"] == {"type": "json_object"}
        assert body["logprobs"] is True and body["top_logprobs"] == 1
        assert result.stop_reason == "length" and result.tokens_prompt == 7
        assert result.logprobs[0]["token"] == "{}"

    def test_a_streamed_tool_turn_is_answered_whole(self):
        call = {"id": "c1", "function": {"name": "w", "arguments": '{"city": "Paris"}'}}
        reply = {
            "choices": [{"message": {"content": "Checking.", "tool_calls": [call]}}],
            "usage": USAGE,
        }
        engine = _engine({"/v1/chat/completions": _Resp(reply)})
        tools = [{"type": "function", "function": {"name": "w"}}]
        stream = engine.chat_stream([ChatMessage("user", "q")], tools=tools)
        chunks = list(stream)
        assert chunks[0] == "Checking."
        assert "Paris" in chunks[1]
        assert (stream.prompt_tokens, stream.completion_tokens) == (7, 2)

    def test_gpt_oss_streams_without_its_reasoning(self):
        lines = _sse(
            {"choices": [{"delta": {"content": "<|channel|>analysis<|message|>hmm<|end|>"}}]},
            {"choices": [{"delta": {"content": ""}}]},
            {"choices": [{"delta": {"content": "<|channel|>final<|message|>42"}}]},
            {"choices": [], "usage": USAGE},
        )
        engine = _engine({"/v1/chat/completions": _Resp(lines=lines)}, _chat_template="<|channel|>")
        stream = engine.chat_stream([ChatMessage("user", "q")])
        assert "".join(stream) == "42"
        assert stream.completion_tokens == 2

    @pytest.mark.parametrize(
        ("fmt", "key", "value"),
        [
            ("json", "json_schema", {}),
            ({"type": "integer"}, "json_schema", {"type": "integer"}),
            ("GBNF:root ::= x", "grammar", "root ::= x"),
        ],
    )
    def test_completion_body_carries_the_format(self, fmt, key, value):
        body = _engine()._completion_body("p", GenerationConfig(response_format=fmt, logprobs=3))
        assert body[key] == value and body["n_probs"] == 3

    def test_generate_with_logprobs(self):
        probs = [{"token": "a", "logprob": -0.1, "top_logprobs": [{"token": "a", "logprob": -0.1}]}]
        reply = {"content": "a", "tokens_predicted": 1, "tokens_evaluated": 3,
                 "stop_type": "limit", "completion_probabilities": probs}  # fmt: skip
        engine = _engine({"/completion": _Resp(reply)})
        result = engine.generate("p", GenerationConfig(logprobs=1))
        assert result.stop_reason == "length" and result.tokens_prompt == 3
        assert result.logprobs[0]["token"] == "a"

    def test_context_tokens_when_tokenize_fails(self, caplog):
        engine = _engine({"/completion": _Resp({"content": "a"}), "/tokenize": _Resp(status=500)})
        with caplog.at_level(logging.WARNING, logger="hfl.engine.llama_server"):
            result = engine.generate("p", GenerationConfig(keep_context=True))
        assert result.context_tokens == []
        assert "keep_context requested but /tokenize failed" in caplog.text

    def test_a_dispatched_chat_keeps_its_reasoning(self):
        import threading

        lines = _sse(
            {"choices": [{"delta": {"reasoning_content": "thinking"}}]},
            {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": USAGE},
        )
        engine = _engine({"/v1/chat/completions": _Resp(lines=lines)})
        with cancel.scope(threading.Event()):
            data = engine._streamed_chat({"messages": []})
        message = data["choices"][0]["message"]
        assert message == {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "thinking",
        }
        assert data["usage"] == USAGE


class TestLora:
    def test_refused_scales_stop_the_process(self, monkeypatch):
        engine = _engine({"/lora-adapters": _Resp(status=400)})
        engine._loras = [("a", "/a.gguf", 0.5)]
        stopped = []
        monkeypatch.setattr(ls, "stop_server", lambda proc: stopped.append(proc))
        client = engine._client
        with pytest.raises(RuntimeError, match="did not take the LoRA scales"):
            engine._set_lora_scales()
        assert client.closed and engine._client is None and stopped == [None]
        sent = client.sent[-1][2]["json"]
        assert sent == [{"id": 0, "scale": 0.5}]

    def test_nothing_to_apply_to(self, tmp_path):
        engine = ls.LlamaServerEngine()
        with pytest.raises(RuntimeError, match="no model loaded"):
            engine.apply_lora(str(tmp_path / "a.gguf"), 1.0)


class TestProperties:
    def test_unloaded(self):
        engine = ls.LlamaServerEngine()
        assert engine.model_name == "" and not engine.is_loaded
        assert engine.generates_on_all_cpu_cores is False
        assert engine.supports_structured_output is True
        assert engine.acceleration is None

    def test_loaded(self):
        engine = _engine(_slots=4, _cpu_only=True)
        assert engine.model_name == "qwen.gguf" and engine.is_loaded
        assert engine.generates_on_all_cpu_cores is True
        assert engine.acceleration == "llama-server · 4 parallel slots"
        assert engine.parallel_slots == 4
