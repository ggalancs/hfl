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
