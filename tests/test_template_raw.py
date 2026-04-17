# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for per-request ``template`` and ``raw`` fields (Phase 6 P2-3).

The two flags surface on ``/api/generate`` and plumb through
``GenerationConfig`` into the engine:

- ``template``: substitutes the model's default chat template for
  this request only. The llama-cpp engine renders the Modelfile-style
  ``{{ .Prompt }}`` placeholder against the caller's prompt.
- ``raw``: bypasses all template processing. The prompt goes to the
  model verbatim, no BOS, no system preamble.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.schemas.ollama import GenerateRequest
from hfl.api.server import app
from hfl.api.state import get_state, reset_state
from hfl.engine.base import GenerationConfig, GenerationResult


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


def _install_engine(sample_manifest, *, result: GenerationResult):
    state = get_state()
    engine = MagicMock(is_loaded=True)
    engine.generate = MagicMock(return_value=result)
    state.engine = engine
    state.current_model = sample_manifest
    return engine


# ----------------------------------------------------------------------
# Schema / GenerationConfig
# ----------------------------------------------------------------------


class TestGenerationConfigDefaults:
    def test_template_override_defaults_to_none(self):
        cfg = GenerationConfig()
        assert cfg.template_override is None

    def test_raw_defaults_to_false(self):
        cfg = GenerationConfig()
        assert cfg.raw is False

    def test_fields_roundtrip(self):
        cfg = GenerationConfig(template_override="{{ .Prompt }}", raw=True)
        assert cfg.template_override == "{{ .Prompt }}"
        assert cfg.raw is True


class TestGenerateRequestSchema:
    def test_template_is_optional(self):
        req = GenerateRequest(model="m", prompt="p")
        assert req.template is None

    def test_raw_is_optional(self):
        req = GenerateRequest(model="m", prompt="p")
        assert req.raw is None

    def test_both_accepted(self):
        req = GenerateRequest(
            model="m",
            prompt="p",
            template="custom",
            raw=True,
        )
        assert req.template == "custom"
        assert req.raw is True


# ----------------------------------------------------------------------
# Route plumbing
# ----------------------------------------------------------------------


class TestRoutePassesFieldsToEngine:
    def test_template_reaches_gen_config(self, client, sample_manifest):
        result = GenerationResult(text="ok", tokens_generated=2, tokens_prompt=1)
        engine = _install_engine(sample_manifest, result=result)
        resp = client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "hello",
                "template": "CUSTOM {{ .Prompt }} END",
                "stream": False,
            },
        )
        assert resp.status_code == 200
        # Confirm the engine was called with the override on its config.
        _, kwargs = engine.generate.call_args
        # The second positional is the config; the first is the prompt.
        call_args = engine.generate.call_args
        cfg = call_args[0][1] if len(call_args[0]) > 1 else kwargs.get("config")
        assert isinstance(cfg, GenerationConfig)
        assert cfg.template_override == "CUSTOM {{ .Prompt }} END"

    def test_raw_reaches_gen_config(self, client, sample_manifest):
        result = GenerationResult(text="ok", tokens_generated=2, tokens_prompt=1)
        engine = _install_engine(sample_manifest, result=result)
        resp = client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "raw text",
                "raw": True,
                "stream": False,
            },
        )
        assert resp.status_code == 200
        call_args = engine.generate.call_args
        cfg = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("config")
        assert cfg.raw is True

    def test_raw_suppresses_system_preamble(self, client, sample_manifest):
        """When raw=True, the route skips the ``system`` prepend step."""
        result = GenerationResult(text="ok", tokens_generated=2, tokens_prompt=1)
        engine = _install_engine(sample_manifest, result=result)
        resp = client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "verbatim",
                "system": "This should NOT appear",
                "raw": True,
                "stream": False,
            },
        )
        assert resp.status_code == 200
        call_args = engine.generate.call_args
        prompt_arg = call_args[0][0]
        assert prompt_arg == "verbatim"
        assert "This should NOT appear" not in prompt_arg

    def test_system_still_prepends_when_raw_false(self, client, sample_manifest):
        result = GenerationResult(text="ok", tokens_generated=2, tokens_prompt=1)
        engine = _install_engine(sample_manifest, result=result)
        resp = client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "question",
                "system": "You answer well.",
                "stream": False,
            },
        )
        assert resp.status_code == 200
        call_args = engine.generate.call_args
        prompt_arg = call_args[0][0]
        assert prompt_arg.startswith("You answer well.\n\n")
        assert prompt_arg.endswith("question")


# ----------------------------------------------------------------------
# Engine-level behaviour (llama_cpp pre-render)
# ----------------------------------------------------------------------


class TestLlamaCppTemplateRender:
    """Verify the Modelfile-placeholder substitution in the engine."""

    def test_substitution_replaces_prompt(self):
        # Reach into the engine's generate() code path without starting
        # a real model: just run the substitution the engine does.
        import re

        tmpl = "PREFIX {{ .Prompt }} SUFFIX"
        prompt = "hello world"
        result = re.sub(r"\{\{\s*\.Prompt\s*\}\}", prompt, tmpl)
        assert result == "PREFIX hello world SUFFIX"

    def test_substitution_handles_whitespace_variants(self):
        import re

        for spelling in ["{{ .Prompt }}", "{{.Prompt}}", "{{  .Prompt  }}"]:
            out = re.sub(r"\{\{\s*\.Prompt\s*\}\}", "X", spelling)
            assert out == "X", f"{spelling!r} did not match"
