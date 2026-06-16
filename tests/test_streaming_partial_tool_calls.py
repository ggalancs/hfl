# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for streaming partial tool calls on /api/chat (Phase 10 P1)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state
from hfl.engine.base import GenerationResult
from hfl.models.manifest import ModelManifest


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


def _install_streaming_engine(manifest, tokens: list[str]):
    state = get_state()
    engine = MagicMock(is_loaded=True)

    def _chat_stream(messages, config, tools=None):
        return iter(tokens)

    engine.chat_stream = MagicMock(side_effect=_chat_stream)
    engine.chat = MagicMock(return_value=GenerationResult(text="".join(tokens)))
    state.engine = engine
    state.current_model = manifest
    return engine


class TestStreamingPartialToolCalls:
    def test_partial_call_appears_on_intermediate_chunk(self, client):
        # Gemma 4 style tool-call: the parser's OpenAI-ish branch sees
        # ``[{"name":"search","arguments":{"q":"cats"}}]`` once complete,
        # and can surface the partial after enough arrives.
        manifest = ModelManifest(
            name="m",
            repo_id="org/m",
            local_path="/tmp/x.gguf",
            format="gguf",
        )
        _install_streaming_engine(
            manifest,
            [
                "Some text before tool call.\n",
                '```tool_call\n{"name": "search",',
                ' "arguments": {"q": "cats"}}\n```',
            ],
        )
        resp = client.post(
            "/api/chat",
            json={
                "model": manifest.name,
                "messages": [{"role": "user", "content": "search cats"}],
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "description": "x",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            },
        )
        assert resp.status_code == 200
        events = [json.loads(line) for line in resp.text.strip().split("\n") if line.strip()]
        # Final event always has done=true.
        assert events[-1]["done"] is True
        # At least one intermediate event carries a non-null tool_calls
        # (the partial surfaces once enough of the JSON is accumulated).
        partials = [
            e for e in events if not e.get("done") and (e.get("message") or {}).get("tool_calls")
        ]
        assert len(partials) >= 1
        # Final envelope's tool_calls matches the parsed call.
        assert events[-1]["message"]["tool_calls"][0]["function"]["name"] == "search"

    def test_no_tool_call_stays_null(self, client):
        manifest = ModelManifest(
            name="n",
            repo_id="org/n",
            local_path="/tmp/x.gguf",
            format="gguf",
        )
        _install_streaming_engine(manifest, ["hello ", "world"])
        resp = client.post(
            "/api/chat",
            json={
                "model": manifest.name,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        events = [json.loads(line) for line in resp.text.strip().split("\n") if line.strip()]
        # Intermediate chunks have ``tool_calls: None`` when no calls
        # are visible.
        inter = [e for e in events if not e.get("done")]
        assert all(
            e["message"]["tool_calls"] is None or e["message"]["tool_calls"] == [] for e in inter
        )


class TestStreamingUsageAndContent:
    """API-5 (streaming done-chunk usage/timings) + API-7 (no tool-call marker
    leakage into Ollama stream content)."""

    def test_chat_done_chunk_has_usage_and_timing(self, client):
        manifest = ModelManifest(name="n", repo_id="org/n", local_path="/tmp/x.gguf", format="gguf")
        _install_streaming_engine(manifest, ["hello ", "world"])
        resp = client.post(
            "/api/chat",
            json={
                "model": manifest.name,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        events = [json.loads(line) for line in resp.text.strip().split("\n") if line.strip()]
        done = events[-1]
        assert done["done"] is True
        # API-5: the fields real Ollama always emits on the final chunk.
        assert done["done_reason"] == "stop"
        assert done["eval_count"] == 2
        for k in (
            "total_duration",
            "eval_duration",
            "prompt_eval_duration",
            "load_duration",
            "prompt_eval_count",
        ):
            assert isinstance(done[k], int) and done[k] >= 0
        # Plain (no-tools) turn streams content verbatim.
        contents = "".join((e.get("message") or {}).get("content") or "" for e in events)
        assert contents == "hello world"

    def test_tool_markers_never_leak_into_content(self, client):
        # Qwen-style markers split across tokens; the per-family parser fires
        # on a model name containing "qwen".
        manifest = ModelManifest(
            name="qwen2.5", repo_id="org/qwen", local_path="/tmp/x.gguf", format="gguf"
        )
        _install_streaming_engine(
            manifest,
            [
                "Let me check. ",
                '<tool_call>\n{"name": "search",',
                ' "arguments": {"q": "cats"}}</tool_call>',
            ],
        )
        resp = client.post(
            "/api/chat",
            json={
                "model": manifest.name,
                "messages": [{"role": "user", "content": "search cats"}],
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "description": "x",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            },
        )
        assert resp.status_code == 200
        events = [json.loads(line) for line in resp.text.strip().split("\n") if line.strip()]
        contents = "".join((e.get("message") or {}).get("content") or "" for e in events)
        # API-7: the raw marker / JSON payload must never appear as content.
        assert "<tool_call>" not in contents
        assert "</tool_call>" not in contents
        assert '"arguments"' not in contents
        # The structured tool call still surfaces on the final chunk.
        assert events[-1]["message"]["tool_calls"][0]["function"]["name"] == "search"
        assert events[-1]["message"]["content"] == ""

    def test_generate_done_chunk_has_usage_and_timing(self, client):
        manifest = ModelManifest(name="g", repo_id="org/g", local_path="/tmp/x.gguf", format="gguf")
        state = get_state()
        engine = MagicMock(is_loaded=True)
        engine.generate_stream = MagicMock(return_value=iter(["foo", "bar"]))
        state.engine = engine
        state.current_model = manifest

        resp = client.post(
            "/api/generate",
            json={"model": manifest.name, "prompt": "hi", "stream": True},
        )
        assert resp.status_code == 200
        events = [json.loads(line) for line in resp.text.strip().split("\n") if line.strip()]
        done = events[-1]
        assert done["done"] is True
        assert done["done_reason"] == "stop"
        assert done["eval_count"] == 2
        assert isinstance(done["total_duration"], int) and done["total_duration"] >= 0
        # Intermediate chunks still carry the response text.
        inter = [e for e in events if not e.get("done")]
        assert "".join(e["response"] for e in inter) == "foobar"
