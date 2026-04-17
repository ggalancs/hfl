# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for Modelfile ``MESSAGE`` few-shot prepending on /api/chat (P3-2).

When a manifest carries ``messages`` entries (produced by ``POST
/api/create`` from a Modelfile with ``MESSAGE`` lines), they are
spliced into every /api/chat request between the system block and
the live turn. This mirrors Ollama's Modelfile semantics: MESSAGE
instructions are canonical few-shot exemplars, not overridable by
the caller.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.routes_native import _baked_messages, _splice_baked_messages
from hfl.api.server import app
from hfl.api.state import get_state, reset_state
from hfl.engine.base import ChatMessage, GenerationResult
from hfl.models.manifest import ModelManifest


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


def _install(manifest, *, result: GenerationResult):
    state = get_state()
    engine = MagicMock(is_loaded=True)
    engine.chat = MagicMock(return_value=result)
    state.engine = engine
    state.current_model = manifest
    return engine


def _manifest_with_messages(name: str, messages: list[dict]) -> ModelManifest:
    return ModelManifest(
        name=name,
        repo_id=f"org/{name}",
        local_path="/tmp/fake.gguf",
        format="gguf",
        messages=messages,
    )


# ----------------------------------------------------------------------
# Helpers (unit-level)
# ----------------------------------------------------------------------


class TestBakedMessagesExtractor:
    def test_none_manifest_gives_empty(self):
        assert _baked_messages(None) == []

    def test_empty_messages_list(self):
        m = _manifest_with_messages("m", [])
        assert _baked_messages(m) == []

    def test_simple_pair(self):
        m = _manifest_with_messages(
            "m",
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ],
        )
        out = _baked_messages(m)
        assert [x.role for x in out] == ["user", "assistant"]
        assert [x.content for x in out] == ["Q", "A"]

    def test_malformed_entries_are_dropped(self):
        m = _manifest_with_messages(
            "m",
            [
                "not-a-dict",  # type: ignore[list-item]
                {"role": 42, "content": "x"},
                {"role": "user"},  # missing content
                {"role": "user", "content": "keep"},
            ],
        )
        out = _baked_messages(m)
        assert len(out) == 1
        assert out[0].content == "keep"


class TestSpliceBakedMessages:
    def test_no_baked_returns_input_unchanged(self):
        req = [ChatMessage(role="user", content="hi")]
        assert _splice_baked_messages(req, []) == req

    def test_inserts_after_leading_system(self):
        req = [
            ChatMessage(role="system", content="S"),
            ChatMessage(role="user", content="live"),
        ]
        baked = [
            ChatMessage(role="user", content="Q"),
            ChatMessage(role="assistant", content="A"),
        ]
        out = _splice_baked_messages(req, baked)
        assert [m.role for m in out] == ["system", "user", "assistant", "user"]
        assert out[1].content == "Q"
        assert out[3].content == "live"

    def test_inserts_at_start_when_no_system(self):
        req = [ChatMessage(role="user", content="live")]
        baked = [ChatMessage(role="user", content="canon")]
        out = _splice_baked_messages(req, baked)
        assert [m.role for m in out] == ["user", "user"]
        assert [m.content for m in out] == ["canon", "live"]

    def test_only_system_input_gets_baked_at_end(self):
        req = [
            ChatMessage(role="system", content="S1"),
            ChatMessage(role="system", content="S2"),
        ]
        baked = [ChatMessage(role="user", content="baked")]
        out = _splice_baked_messages(req, baked)
        assert [m.role for m in out] == ["system", "system", "user"]


# ----------------------------------------------------------------------
# End-to-end route integration
# ----------------------------------------------------------------------


class TestChatRouteSplicesBaked:
    def test_baked_messages_spliced_before_live_turn(self, client):
        m = _manifest_with_messages(
            "coder",
            [
                {"role": "user", "content": "How do I sort?"},
                {"role": "assistant", "content": "Use sorted()."},
            ],
        )
        engine = _install(
            m, result=GenerationResult(text="ok", tokens_generated=1, tokens_prompt=1)
        )

        resp = client.post(
            "/api/chat",
            json={
                "model": m.name,
                "messages": [
                    {"role": "system", "content": "You are a coder."},
                    {"role": "user", "content": "What about reverse?"},
                ],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        passed = engine.chat.call_args[0][0]
        assert [x.role for x in passed] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert passed[1].content == "How do I sort?"
        assert passed[3].content == "What about reverse?"

    def test_no_baked_leaves_request_unchanged(self, client):
        m = _manifest_with_messages("bare", [])
        engine = _install(
            m, result=GenerationResult(text="ok", tokens_generated=1, tokens_prompt=1)
        )
        resp = client.post(
            "/api/chat",
            json={
                "model": m.name,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        passed = engine.chat.call_args[0][0]
        assert len(passed) == 1
        assert passed[0].content == "hi"
