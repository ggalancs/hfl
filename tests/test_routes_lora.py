# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Integration tests for ``/api/lora/*`` endpoints (V4 F4)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state
from hfl.engine.lora import reset_registry


@pytest.fixture(autouse=True)
def fresh_lora_registry():
    reset_registry()
    yield
    reset_registry()


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


@pytest.fixture
def adapter_file(tmp_path):
    p = tmp_path / "adapter.safetensors"
    p.write_bytes(b"\x00" * 32)
    return str(p)


def _wire_engine(manifest):
    state = get_state()
    engine = MagicMock(spec=["apply_lora", "remove_lora", "is_loaded"])
    engine.is_loaded = True
    engine.apply_lora = MagicMock()
    engine.remove_lora = MagicMock()
    state.engine = engine
    state.current_model = manifest
    return engine


class TestApplyLoraRoute:
    def test_apply_returns_adapter_info(self, client, llm_manifest, adapter_file):
        _wire_engine(llm_manifest)
        response = client.post(
            "/api/lora/apply",
            json={"model": llm_manifest.name, "lora_path": adapter_file, "scale": 0.6},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["path"] == adapter_file
        assert body["scale"] == 0.6
        assert "adapter_id" in body

    def test_missing_path_returns_404(self, client, llm_manifest, tmp_path):
        _wire_engine(llm_manifest)
        ghost = tmp_path / "ghost.safetensors"
        response = client.post(
            "/api/lora/apply",
            json={"model": llm_manifest.name, "lora_path": str(ghost)},
        )
        assert response.status_code == 404

    def test_invalid_scale_returns_400(self, client, llm_manifest, adapter_file):
        _wire_engine(llm_manifest)
        response = client.post(
            "/api/lora/apply",
            json={"model": llm_manifest.name, "lora_path": adapter_file, "scale": 99.0},
        )
        # FastAPI/Pydantic rejects scale > 5.0 at the schema layer.
        assert response.status_code in (400, 422)

    def test_unsupported_engine_returns_503(self, client, llm_manifest, adapter_file):
        state = get_state()
        engine = MagicMock(spec=[])  # no apply_lora path at all
        engine.is_loaded = True
        state.engine = engine
        state.current_model = llm_manifest

        response = client.post(
            "/api/lora/apply",
            json={"model": llm_manifest.name, "lora_path": adapter_file},
        )
        assert response.status_code == 503


class TestRemoveLoraRoute:
    def test_remove_known_adapter(self, client, llm_manifest, adapter_file):
        _wire_engine(llm_manifest)
        applied = client.post(
            "/api/lora/apply",
            json={"model": llm_manifest.name, "lora_path": adapter_file},
        ).json()

        response = client.post(
            "/api/lora/remove",
            json={"model": llm_manifest.name, "adapter_id": applied["adapter_id"]},
        )
        assert response.status_code == 200
        assert response.json()["removed"] is True

    def test_remove_unknown_returns_404(self, client, llm_manifest):
        _wire_engine(llm_manifest)
        response = client.post(
            "/api/lora/remove",
            json={"model": llm_manifest.name, "adapter_id": "never-was"},
        )
        assert response.status_code == 404


class TestListLoraRoute:
    def test_global_listing(self, client, llm_manifest, adapter_file):
        _wire_engine(llm_manifest)
        client.post(
            "/api/lora/apply",
            json={"model": llm_manifest.name, "lora_path": adapter_file, "name": "code"},
        )
        body = client.get("/api/lora").json()
        assert len(body["adapters"]) == 1
        assert body["adapters"][0]["name"] == "code"

    def test_per_model_listing(self, client, llm_manifest, adapter_file):
        _wire_engine(llm_manifest)
        client.post(
            "/api/lora/apply",
            json={"model": llm_manifest.name, "lora_path": adapter_file},
        )
        body = client.get(f"/api/lora/{llm_manifest.name}").json()
        assert body["model"] == llm_manifest.name
        assert len(body["adapters"]) == 1
