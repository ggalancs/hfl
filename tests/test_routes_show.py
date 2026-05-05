# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for Ollama-compatible ``POST /api/show``.

The envelope is a hard contract: Open WebUI and ``ollama-python``
read specific keys (``modelfile``, ``parameters``, ``template``,
``details``, ``model_info``, ``capabilities``, ``license``) — missing
or mistyped fields cause silent UI breakage.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.converter.modelfile import render_modelfile


@pytest.fixture
def client(temp_config):
    return TestClient(app)


@pytest.fixture
def registered_llm(sample_manifest):
    """Put ``sample_manifest`` into the registry under a known name."""
    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    sample_manifest.name = "qwen-coder:7b"
    sample_manifest.architecture = "qwen"
    sample_manifest.parameters = "7B"
    sample_manifest.quantization = "Q4_K_M"
    sample_manifest.format = "gguf"
    sample_manifest.context_length = 32768
    sample_manifest.chat_template = "<|im_start|>{{role}}\n{{content}}<|im_end|>"
    sample_manifest.license = "Apache-2.0"
    sample_manifest.license_name = "Apache License 2.0"
    registry.add(sample_manifest)
    yield sample_manifest


class TestRoutesShowHappy:
    def test_returns_full_envelope(self, client, registered_llm):
        response = client.post("/api/show", json={"model": "qwen-coder:7b"})
        assert response.status_code == 200
        body = response.json()
        # Contract fields — every one of these is keyed on by a
        # downstream client.
        required = {
            "modelfile",
            "parameters",
            "template",
            "details",
            "model_info",
            "capabilities",
            "license",
        }
        assert required <= set(body.keys()), (
            f"Missing contract fields: {required - set(body.keys())}"
        )

    def test_modelfile_is_renderable(self, client, registered_llm):
        """``modelfile`` must be a non-empty string that starts with FROM."""
        body = client.post("/api/show", json={"model": "qwen-coder:7b"}).json()
        assert isinstance(body["modelfile"], str)
        assert body["modelfile"].startswith("FROM ")

    def test_parameters_format_is_multiline_ollama_style(self, client, registered_llm):
        """``parameters`` is one "key value" per line, Ollama-style."""
        body = client.post("/api/show", json={"model": "qwen-coder:7b"}).json()
        params = body["parameters"]
        assert "num_ctx 32768" in params
        # No PARAMETER keyword in the /api/show output (unlike Modelfile)
        assert "PARAMETER" not in params

    def test_template_is_the_manifest_chat_template(self, client, registered_llm):
        body = client.post("/api/show", json={"model": "qwen-coder:7b"}).json()
        assert body["template"] == registered_llm.chat_template

    def test_details_shape_matches_ps(self, client, registered_llm):
        body = client.post("/api/show", json={"model": "qwen-coder:7b"}).json()
        details = body["details"]
        assert details["format"] == "gguf"
        assert details["family"] == "qwen"
        assert details["parameter_size"] == "7B"
        assert details["quantization_level"] == "Q4_K_M"

    def test_model_info_uses_gguf_key_style(self, client, registered_llm):
        """Typed wrappers look for ``general.architecture`` etc."""
        body = client.post("/api/show", json={"model": "qwen-coder:7b"}).json()
        info = body["model_info"]
        assert info.get("general.architecture") == "qwen"
        assert info.get("qwen.context_length") == 32768
        assert info.get("general.quantization") == "Q4_K_M"

    def test_capabilities_includes_tools_for_qwen(self, client, registered_llm):
        body = client.post("/api/show", json={"model": "qwen-coder:7b"}).json()
        assert "completion" in body["capabilities"]
        assert "tools" in body["capabilities"]
        assert "insert" in body["capabilities"]  # qwen-coder → FIM

    def test_license_surfaced(self, client, registered_llm):
        body = client.post("/api/show", json={"model": "qwen-coder:7b"}).json()
        assert body["license"] == "Apache License 2.0"


class TestRoutesShowVerbose:
    def test_verbose_adds_extra_key(self, client, registered_llm):
        body = client.post("/api/show", json={"model": "qwen-coder:7b", "verbose": True}).json()
        # Non-verbose path does NOT carry the hfl.verbose marker
        assert body["model_info"].get("hfl.verbose") is True

    def test_non_verbose_omits_the_marker(self, client, registered_llm):
        body = client.post("/api/show", json={"model": "qwen-coder:7b"}).json()
        assert "hfl.verbose" not in body["model_info"]


class TestRoutesShowErrors:
    def test_unknown_model_returns_404(self, client):
        response = client.post("/api/show", json={"model": "does-not-exist"})
        assert response.status_code == 404
        body = response.json()
        # R10 envelope: {"error": "...", "code": "ModelNotFoundError"}
        assert body.get("code") == "ModelNotFoundError"

    def test_empty_model_name_returns_422(self, client):
        """Pydantic ``min_length=1`` guards the name field."""
        response = client.post("/api/show", json={"model": ""})
        assert response.status_code == 422

    def test_missing_body_returns_422(self, client):
        response = client.post("/api/show", json={})
        assert response.status_code == 422


class TestRoutesShowParametersRendering:
    """Coverage for the ``default_parameters`` rendering branches in
    ``_render_parameters_block`` — V4 audit closeout."""

    def test_default_parameters_emit_one_line_per_key(self, client, sample_manifest):
        """Non-stop parameters should render as ``"<key> <value>"``,
        sorted by key for stable diffs."""
        from hfl.models.registry import ModelRegistry

        sample_manifest.name = "with-params"
        sample_manifest.architecture = "qwen"
        sample_manifest.context_length = 4096
        sample_manifest.default_parameters = {
            "temperature": 0.7,
            "top_p": 0.95,
            "num_predict": 256,
        }
        ModelRegistry().add(sample_manifest)

        body = client.post("/api/show", json={"model": "with-params"}).json()
        params = body["parameters"]
        # All three keys appear, one per line, in alphabetical order.
        lines = [line for line in params.splitlines() if line]
        assert lines[0].startswith("num_ctx ")
        non_ctx = lines[1:]
        assert non_ctx == sorted(non_ctx)
        assert "temperature 0.7" in params
        assert "top_p 0.95" in params
        assert "num_predict 256" in params

    def test_stop_list_emits_one_quoted_line_per_value(self, client, sample_manifest):
        """``stop`` is special: each list entry becomes its own
        ``stop "<escaped>"`` line."""
        from hfl.models.registry import ModelRegistry

        sample_manifest.name = "with-stops"
        sample_manifest.architecture = "qwen"
        sample_manifest.default_parameters = {
            "stop": ["<|im_end|>", "</s>", 'with "quotes"'],
        }
        ModelRegistry().add(sample_manifest)

        body = client.post("/api/show", json={"model": "with-stops"}).json()
        params = body["parameters"]
        assert 'stop "<|im_end|>"' in params
        assert 'stop "</s>"' in params
        # Internal quotes are escaped.
        assert r'stop "with \"quotes\""' in params

    def test_stop_scalar_value_is_normalised_to_one_line(self, client, sample_manifest):
        """A non-list ``stop`` value still renders as a single quoted
        line — ``isinstance(..., (list, tuple))`` short-circuits."""
        from hfl.models.registry import ModelRegistry

        sample_manifest.name = "with-stop-scalar"
        sample_manifest.architecture = "qwen"
        sample_manifest.default_parameters = {"stop": "<|eot|>"}
        ModelRegistry().add(sample_manifest)

        body = client.post("/api/show", json={"model": "with-stop-scalar"}).json()
        assert 'stop "<|eot|>"' in body["parameters"]


class TestRoutesShowModelInfoDigest:
    def test_file_hash_surfaces_as_general_digest(self, client, sample_manifest):
        """When the manifest carries a ``file_hash``, ``model_info``
        exposes it as ``general.digest`` (Ollama convention)."""
        from hfl.models.registry import ModelRegistry

        sample_manifest.name = "with-hash"
        sample_manifest.architecture = "qwen"
        sample_manifest.file_hash = "sha256:abc123"
        ModelRegistry().add(sample_manifest)

        body = client.post("/api/show", json={"model": "with-hash"}).json()
        assert body["model_info"]["general.digest"] == "sha256:abc123"


class TestRenderModelfile:
    """Unit tests for the Modelfile renderer itself."""

    def test_minimal_manifest_emits_from_line(self, registered_llm):
        text = render_modelfile(registered_llm)
        assert text.startswith("FROM ")
        assert text.endswith("\n")

    def test_template_block_quoted_with_triple_quotes(self, registered_llm):
        text = render_modelfile(registered_llm)
        assert 'TEMPLATE """' in text
        assert '"""' in text.split('TEMPLATE """', 1)[1]

    def test_parameter_num_ctx_rendered(self, registered_llm):
        text = render_modelfile(registered_llm)
        assert "PARAMETER num_ctx 32768" in text

    def test_license_block_rendered(self, registered_llm):
        text = render_modelfile(registered_llm)
        assert 'LICENSE """Apache License 2.0"""' in text

    def test_stop_strings_escape_quotes_and_backslashes(self, sample_manifest):
        """Special chars in stop values round-trip through the escaper."""
        sample_manifest.default_parameters = {"stop": ['he said "hi"', "back\\slash"]}
        text = render_modelfile(sample_manifest)
        assert 'PARAMETER stop "he said \\"hi\\""' in text
        assert 'PARAMETER stop "back\\\\slash"' in text

    def test_from_uses_file_hash_when_present(self, sample_manifest):
        sample_manifest.file_hash = "sha256:" + "a" * 64
        text = render_modelfile(sample_manifest)
        assert f"FROM {sample_manifest.file_hash}" in text

    def test_from_falls_back_to_local_path(self, sample_manifest):
        sample_manifest.file_hash = None
        sample_manifest.local_path = "/tmp/model.gguf"
        text = render_modelfile(sample_manifest)
        assert "FROM /tmp/model.gguf" in text

    def test_render_is_deterministic(self, registered_llm):
        """Twice in a row → byte-identical output (snapshot stability)."""
        assert render_modelfile(registered_llm) == render_modelfile(registered_llm)
