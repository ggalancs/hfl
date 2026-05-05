# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Integration tests for the V4 F6 snapshot endpoints."""

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
    )


class _PickleableState:
    """Plain pickleable stand-in for ``Llama.save_state()``."""

    def __init__(self, n_tokens: int = 10):
        self.n_tokens = n_tokens
        self.payload = b"\x00" * 8


def _wire_engine(manifest):
    state = get_state()
    engine = MagicMock(spec=["save_state", "load_state", "is_loaded"])
    engine.is_loaded = True
    engine.save_state = MagicMock(return_value=_PickleableState(n_tokens=10))
    engine.load_state = MagicMock()
    state.engine = engine
    state.current_model = manifest
    return engine


class TestSnapshotSave:
    def test_save_returns_meta(self, client, llm_manifest):
        _wire_engine(llm_manifest)
        response = client.post(
            "/api/snapshot/save",
            json={"model": llm_manifest.name, "name": "warm-1"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "warm-1"
        assert body["model"] == llm_manifest.name
        assert body["tokens"] == 10
        assert body["bytes"] > 0

    def test_save_with_invalid_name_returns_400(self, client, llm_manifest):
        _wire_engine(llm_manifest)
        response = client.post(
            "/api/snapshot/save",
            json={"model": llm_manifest.name, "name": "../escape"},
        )
        assert response.status_code == 400

    def test_save_unsupported_engine_returns_503(self, client, llm_manifest):
        state = get_state()
        engine = MagicMock(spec=[])  # no save_state
        engine.is_loaded = True
        state.engine = engine
        state.current_model = llm_manifest

        response = client.post(
            "/api/snapshot/save",
            json={"model": llm_manifest.name, "name": "warm-1"},
        )
        assert response.status_code == 503


class TestSnapshotLoad:
    def test_round_trip(self, client, llm_manifest):
        _wire_engine(llm_manifest)
        # Save
        client.post(
            "/api/snapshot/save",
            json={"model": llm_manifest.name, "name": "warm-1"},
        )
        # Load
        response = client.post(
            "/api/snapshot/load",
            json={"model": llm_manifest.name, "name": "warm-1"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "warm-1"

    def test_load_missing_returns_404(self, client, llm_manifest):
        _wire_engine(llm_manifest)
        response = client.post(
            "/api/snapshot/load",
            json={"model": llm_manifest.name, "name": "never-existed"},
        )
        assert response.status_code == 404

    def test_load_wrong_model_returns_400(self, client, llm_manifest, monkeypatch):
        """Saved for model A, attempted load into model B → 400 with
        the cross-model error from the snapshot layer."""
        from hfl.models.manifest import ModelManifest

        manifest_a = ModelManifest(
            name="model-a",
            repo_id="org/a",
            local_path="/tmp/a",
            format="gguf",
            architecture="qwen",
            parameters="7B",
        )
        manifest_b = ModelManifest(
            name="model-b",
            repo_id="org/b",
            local_path="/tmp/b",
            format="gguf",
            architecture="qwen",
            parameters="7B",
        )

        # Wire for A, save.
        _wire_engine(manifest_a)
        client.post(
            "/api/snapshot/save",
            json={"model": "model-a", "name": "warm-1"},
        )
        # Re-wire for B, attempt load.
        _wire_engine(manifest_b)
        response = client.post(
            "/api/snapshot/load",
            json={"model": "model-b", "name": "warm-1"},
        )
        assert response.status_code == 400
        assert "model-a" in response.json()["detail"]


class TestSnapshotListAndDelete:
    def test_list_after_save(self, client, llm_manifest):
        _wire_engine(llm_manifest)
        client.post(
            "/api/snapshot/save",
            json={"model": llm_manifest.name, "name": "warm-1"},
        )
        response = client.get("/api/snapshot")
        assert response.status_code == 200
        names = [s["name"] for s in response.json()["snapshots"]]
        assert "warm-1" in names

    def test_delete_existing(self, client, llm_manifest):
        _wire_engine(llm_manifest)
        client.post(
            "/api/snapshot/save",
            json={"model": llm_manifest.name, "name": "warm-1"},
        )
        response = client.delete("/api/snapshot/warm-1")
        assert response.status_code == 200
        assert response.json()["deleted"] is True

    def test_delete_missing_returns_404(self, client):
        response = client.delete("/api/snapshot/never-existed")
        assert response.status_code == 404

    def test_delete_invalid_name_returns_400(self, client):
        response = client.delete("/api/snapshot/..")
        # Invalid name surfaces as ValueError → 400 OR FastAPI may
        # reject the path char first as 404. Both indicate "not
        # accepted".
        assert response.status_code in (400, 404)
