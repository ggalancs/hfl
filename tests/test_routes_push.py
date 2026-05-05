# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ``POST /api/push`` (HF Hub uploader).

The HF SDK is patched out in every case — these tests pin the request
parsing, the manifest resolution, the NDJSON event grammar and the
auth fallback chain, all without touching the network.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import reset_state


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


@pytest.fixture
def model_dir(tmp_path):
    """A minimal model directory with two artefact files."""
    d = tmp_path / "qwen-coder-7b"
    d.mkdir()
    (d / "model.gguf").write_bytes(b"GGUF" + b"\x00" * 1024)
    (d / "tokenizer.json").write_text("{}")
    return d


@pytest.fixture
def registered_model(model_dir):
    """Inject a manifest into the singleton registry."""
    from hfl.core.container import get_registry
    from hfl.models.manifest import ModelManifest

    manifest = ModelManifest(
        name="qwen-coder-7b",
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        local_path=str(model_dir),
        format="gguf",
        architecture="qwen",
        parameters="7B",
    )
    registry = get_registry()
    registry.add(manifest)
    yield manifest
    registry.remove(manifest.name)


@pytest.fixture
def fake_hf_api(monkeypatch):
    """Replace ``HfApi`` so no network hit happens during tests."""
    api = MagicMock()
    api.create_repo = MagicMock(return_value=None)
    commit = MagicMock()
    commit.commit_url = "https://huggingface.co/u/m/commit/abc"
    api.upload_folder = MagicMock(return_value=commit)

    from hfl.api import routes_push as module

    monkeypatch.setattr(module, "_build_api", lambda: api)
    return api


class TestPushHappyPath:
    def test_streaming_emits_full_event_grammar(self, client, registered_model, fake_hf_api):
        response = client.post(
            "/api/push",
            json={
                "model": registered_model.name,
                "destination": "user/qwen-clone",
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")

        events = [json.loads(line) for line in response.text.splitlines() if line]
        statuses = [e["status"] for e in events]
        # Required prefix per the route docstring.
        assert statuses[0] == "preparing"
        assert "ensuring repository" in statuses
        assert "uploading" in statuses
        assert statuses[-1] == "success"

        # Last event also exposes the commit URL when the API returns one.
        assert events[-1]["commit_url"].endswith("/commit/abc")

    def test_non_stream_returns_last_event_as_json(self, client, registered_model, fake_hf_api):
        response = client.post(
            "/api/push",
            json={
                "model": registered_model.name,
                "destination": "user/qwen-clone",
                "stream": False,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["repo"] == "user/qwen-clone"

    def test_ollama_name_field_is_accepted_alias(self, client, registered_model, fake_hf_api):
        """Ollama clients send ``name`` rather than ``destination``;
        both must work."""
        response = client.post(
            "/api/push",
            json={
                "model": registered_model.name,
                "name": "user/qwen-clone",
                "stream": False,
            },
        )
        assert response.status_code == 200
        assert response.json()["repo"] == "user/qwen-clone"


class TestPushAuth:
    def test_explicit_token_is_passed_to_hf_api(self, client, registered_model, fake_hf_api):
        client.post(
            "/api/push",
            json={
                "model": registered_model.name,
                "destination": "user/qwen-clone",
                "token": "hf_explicit",
                "stream": False,
            },
        )
        # ``create_repo`` is called with the explicit token.
        kwargs = fake_hf_api.create_repo.call_args.kwargs
        assert kwargs["token"] == "hf_explicit"

    def test_falls_back_to_config_token(self, client, registered_model, fake_hf_api, monkeypatch):
        from hfl.config import config

        monkeypatch.setattr(config, "hf_token", "hf_from_config", raising=False)
        client.post(
            "/api/push",
            json={
                "model": registered_model.name,
                "destination": "user/qwen-clone",
                "stream": False,
            },
        )
        kwargs = fake_hf_api.upload_folder.call_args.kwargs
        assert kwargs["token"] == "hf_from_config"


class TestPushValidation:
    def test_unknown_model_returns_404(self, client, fake_hf_api):
        response = client.post(
            "/api/push",
            json={"model": "does-not-exist", "destination": "x/y", "stream": False},
        )
        assert response.status_code == 404

    def test_missing_destination_returns_400(self, client, registered_model, fake_hf_api):
        response = client.post(
            "/api/push",
            json={"model": registered_model.name, "stream": False},
        )
        assert response.status_code == 400

    def test_malformed_destination_returns_400(self, client, registered_model, fake_hf_api):
        response = client.post(
            "/api/push",
            json={
                "model": registered_model.name,
                "destination": "no-namespace",
                "stream": False,
            },
        )
        assert response.status_code == 400


class TestPushFailureSurfaces:
    def test_hub_failure_emits_failed_event(self, client, registered_model, fake_hf_api):
        fake_hf_api.upload_folder.side_effect = RuntimeError("hub said nope")

        response = client.post(
            "/api/push",
            json={
                "model": registered_model.name,
                "destination": "user/qwen-clone",
                "stream": True,
            },
        )
        assert response.status_code == 200  # NDJSON itself succeeds
        events = [json.loads(line) for line in response.text.splitlines() if line]
        assert events[-1]["status"] == "failed"
        assert "hub said nope" in events[-1]["error"]
