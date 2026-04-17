# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""HTTP contract tests for ``POST /api/create`` (Phase 6 P2-1)."""

from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import reset_state
from hfl.hub.blobs import blob_path
from hfl.models.manifest import ModelManifest
from hfl.models.registry import get_registry, reset_registry


@pytest.fixture
def client(temp_config):
    reset_state()
    reset_registry()
    yield TestClient(app)
    reset_state()


def _make_parent_manifest(temp_config, name: str = "llama3.3") -> ModelManifest:
    """Write a fake GGUF on disk and register it so FROM can resolve."""
    gguf = temp_config.home_dir / "models" / f"{name}.gguf"
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(b"fake gguf content")
    manifest = ModelManifest(
        name=name,
        repo_id=f"org/{name}",
        local_path=str(gguf),
        format="gguf",
        size_bytes=gguf.stat().st_size,
        file_hash=hashlib.sha256(gguf.read_bytes()).hexdigest(),
    )
    get_registry().add(manifest)
    return manifest


# ----------------------------------------------------------------------
# Streaming (default)
# ----------------------------------------------------------------------


class TestCreateStreamingFromModelName:
    def test_creates_derived_manifest_with_system(self, client, temp_config):
        parent = _make_parent_manifest(temp_config)
        body = f'FROM {parent.name}\nSYSTEM """You are a coder."""\nPARAMETER num_ctx 4096\n'
        resp = client.post(
            "/api/create",
            json={"model": "coder", "modelfile": body, "stream": True},
        )
        assert resp.status_code == 200

        events = [json.loads(line) for line in resp.text.strip().split("\n") if line.strip()]
        # First event is always parsing; last event is success.
        assert events[0] == {"status": "parsing modelfile"}
        assert events[-1] == {"status": "success"}
        # Pipeline includes creating + writing stages.
        statuses = {e["status"] for e in events if "status" in e}
        assert "creating model" in statuses
        assert "writing manifest" in statuses

        # Registry should now have the derived manifest.
        derived = get_registry().get("coder")
        assert derived is not None
        assert derived.parent_name == parent.name
        assert derived.parent_digest == parent.file_hash
        assert derived.system == "You are a coder."
        assert derived.context_length == 4096
        # Derived model points at the same GGUF blob.
        assert derived.local_path == parent.local_path

    def test_multiple_params_preserved(self, client, temp_config):
        _make_parent_manifest(temp_config)
        body = (
            "FROM llama3.3\n"
            "PARAMETER temperature 0.7\n"
            "PARAMETER top_k 40\n"
            'PARAMETER stop "<|eot_id|>"\n'
        )
        resp = client.post(
            "/api/create",
            json={"model": "tuned", "modelfile": body, "stream": True},
        )
        assert resp.status_code == 200
        derived = get_registry().get("tuned")
        assert derived is not None
        assert derived.default_parameters["temperature"] == 0.7
        assert derived.default_parameters["top_k"] == 40
        assert derived.default_parameters["stop"] == ["<|eot_id|>"]


# ----------------------------------------------------------------------
# Streaming — FROM blob
# ----------------------------------------------------------------------


class TestCreateFromBlob:
    def test_creates_from_uploaded_blob(self, client, temp_config):
        # Upload a blob first.
        data = b"gguf bytes here"
        digest = hashlib.sha256(data).hexdigest()
        upload = client.post(f"/api/blobs/sha256:{digest}", content=data)
        assert upload.status_code == 201

        body = f"FROM sha256:{digest}\n"
        resp = client.post(
            "/api/create",
            json={"model": "fromblob", "modelfile": body, "stream": True},
        )
        assert resp.status_code == 200
        events = [json.loads(line) for line in resp.text.strip().split("\n") if line.strip()]
        assert any("using existing layer sha256:" in (e.get("status") or "") for e in events)
        assert events[-1] == {"status": "success"}
        derived = get_registry().get("fromblob")
        assert derived is not None
        assert derived.parent_digest == f"sha256:{digest}"
        assert str(blob_path(digest)) == derived.local_path

    def test_unknown_blob_fails_with_error_event(self, client):
        missing = "f" * 64
        body = f"FROM sha256:{missing}\n"
        resp = client.post(
            "/api/create",
            json={"model": "nobody", "modelfile": body, "stream": True},
        )
        assert resp.status_code == 200  # stream body is always 200
        events = [json.loads(line) for line in resp.text.strip().split("\n") if line.strip()]
        assert any("error" in e for e in events)
        err = [e for e in events if "error" in e][0]
        # Curated error message, not leaking the specific digest
        # (CodeQL py/stack-trace-exposure posture — 0.12.2+).
        assert "FROM" in err["error"]


# ----------------------------------------------------------------------
# Non-streaming
# ----------------------------------------------------------------------


class TestCreateNonStreaming:
    def test_returns_single_envelope_on_success(self, client, temp_config):
        _make_parent_manifest(temp_config)
        resp = client.post(
            "/api/create",
            json={
                "model": "nonstream",
                "modelfile": "FROM llama3.3\n",
                "stream": False,
            },
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "success"}

    def test_returns_400_envelope_on_error(self, client):
        resp = client.post(
            "/api/create",
            json={
                "model": "missing",
                "modelfile": "FROM ghost-model\n",
                "stream": False,
            },
        )
        assert resp.status_code == 400
        assert "error" in resp.json()


# ----------------------------------------------------------------------
# Structured fields
# ----------------------------------------------------------------------


class TestStructuredFields:
    def test_from_field_without_modelfile_body(self, client, temp_config):
        _make_parent_manifest(temp_config)
        resp = client.post(
            "/api/create",
            json={
                "model": "structured",
                "from": "llama3.3",
                "system": "You are helpful.",
                "stream": False,
            },
        )
        assert resp.status_code == 200
        derived = get_registry().get("structured")
        assert derived is not None
        assert derived.system == "You are helpful."

    def test_structured_overrides_modelfile_body(self, client, temp_config):
        _make_parent_manifest(temp_config)
        resp = client.post(
            "/api/create",
            json={
                "model": "override",
                "modelfile": 'FROM llama3.3\nSYSTEM """old"""\n',
                "system": "new",
                "stream": False,
            },
        )
        assert resp.status_code == 200
        derived = get_registry().get("override")
        assert derived is not None
        assert derived.system == "new"

    def test_missing_from_fails(self, client):
        resp = client.post(
            "/api/create",
            json={
                "model": "nofrom",
                "modelfile": "PARAMETER num_ctx 4096\n",
                "stream": False,
            },
        )
        assert resp.status_code == 400
        # Post-0.12.2: curated generic message, precise detail logs.
        assert "modelfile" in resp.json()["error"].lower()

    def test_invalid_modelfile_syntax_fails(self, client):
        resp = client.post(
            "/api/create",
            json={
                "model": "bad",
                "modelfile": "FROM x\nWEIRD instruction\n",
                "stream": False,
            },
        )
        assert resp.status_code == 400
        # Curated error, precise detail in server log.
        assert "modelfile parse error" in resp.json()["error"].lower()


# ----------------------------------------------------------------------
# Messages
# ----------------------------------------------------------------------


class TestMessagesPersistence:
    def test_modelfile_message_persisted(self, client, temp_config):
        _make_parent_manifest(temp_config)
        body = 'FROM llama3.3\nMESSAGE user "hello"\nMESSAGE assistant "hi there"\n'
        resp = client.post(
            "/api/create",
            json={"model": "withmsg", "modelfile": body, "stream": False},
        )
        assert resp.status_code == 200
        derived = get_registry().get("withmsg")
        assert derived is not None
        assert derived.messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
