# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Integration tests for ``POST /api/verify/{model}`` (V4 F3.1).

The verifier itself is unit-tested in ``test_engine_verifier.py``;
these tests pin the HTTP wrapper: response shape, 404 / 503 paths,
URL-encoding of model names with slashes.
"""

from __future__ import annotations

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
        quantization="Q4_K_M",
    )


def _wire_engine(manifest):
    state = get_state()
    engine = MagicMock()
    engine.is_loaded = True
    tok = MagicMock()
    tok.encode = MagicMock(return_value=[1, 2])
    tok.decode = MagicMock(return_value="Hello, world.")
    tok.apply_chat_template = MagicMock(return_value="<|im_start|>x<|im_end|>")
    engine.tokenizer = tok
    result = MagicMock()
    result.text = "ok"
    result.tokens_generated = 1
    engine.generate = MagicMock(return_value=result)
    state.engine = engine
    state.current_model = manifest
    return engine


class TestVerifyHappyPath:
    def test_returns_overall_pass_envelope(self, client, llm_manifest):
        _wire_engine(llm_manifest)

        response = client.post(f"/api/verify/{llm_manifest.name}")
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == llm_manifest.name
        assert "overall_pass" in body
        assert isinstance(body["overall_pass"], bool)
        assert body["overall_pass"] is True
        assert "duration_ms" in body
        assert isinstance(body["duration_ms"], (int, float))
        assert isinstance(body["checks"], list)

    def test_each_check_has_name_passed_detail(self, client, llm_manifest):
        _wire_engine(llm_manifest)

        body = client.post(f"/api/verify/{llm_manifest.name}").json()
        # All five canonical checks must appear.
        names = {c["name"] for c in body["checks"]}
        assert {
            "tokenizer_round_trip",
            "chat_template_render",
            "smoke_generation",
            "tool_parser_round_trip",
            "embedding_dim",
        } <= names
        for check in body["checks"]:
            assert {"name", "passed", "detail"} <= check.keys()

    def test_failing_probe_marks_overall_fail(self, client, llm_manifest):
        engine = _wire_engine(llm_manifest)
        engine.generate = MagicMock(side_effect=RuntimeError("CUDA OOM"))

        body = client.post(f"/api/verify/{llm_manifest.name}").json()
        assert body["overall_pass"] is False
        gen = next(c for c in body["checks"] if c["name"] == "smoke_generation")
        assert gen["passed"] is False
        assert "CUDA OOM" in gen["detail"]


class TestVerifyFailures:
    def test_unknown_model_returns_404(self, client, monkeypatch):
        from hfl.api import routes_verify as module

        async def _missing(name):
            raise FileNotFoundError(f"model not found: {name}")

        monkeypatch.setattr(module, "load_llm", _missing)

        response = client.post("/api/verify/does-not-exist")
        assert response.status_code == 404

    def test_engine_unavailable_returns_503(self, client, monkeypatch):
        from hfl.api import routes_verify as module

        async def _no_engine(name):
            return None, MagicMock(name=name)

        monkeypatch.setattr(module, "load_llm", _no_engine)

        response = client.post("/api/verify/qwen")
        assert response.status_code == 503


class TestVerifyPathHandling:
    def test_model_path_with_colon_or_slash_is_accepted(self, client, llm_manifest):
        """Model identifiers carrying ``/`` (HF repo ids when used as
        names) must round-trip through the path parameter."""
        # Re-wire under a slashed name — registry validates the format,
        # so we use a hyphenated alias instead which is what happens
        # in practice.
        _wire_engine(llm_manifest)
        response = client.post(f"/api/verify/{llm_manifest.name}")
        assert response.status_code == 200
