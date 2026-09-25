# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Short names over the API: what Ollama clients (Open WebUI...) send.

``/api/pull {"model": "llama3.2"}`` pulls the best GGUF build and registers
it under that name, so ``/api/chat {"model": "llama3.2"}`` then works; a
tagged name (``qwen3:8b``) is kept as ``qwen3-8b`` (':' is not allowed in a
name) and found again by the chat routes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hfl.hub.shortname import ShortNameMatch


@pytest.fixture
def owner(temp_config):
    from hfl.api.server import app

    return TestClient(app, client=("127.0.0.1", 50000))


def _choice(name="llama3.2"):
    return ShortNameMatch(
        name=name,
        repo_id="unsloth/Llama-3.2-3B-Instruct-GGUF",
        quantization="Q4_K_M",
        size_bytes=2_000_000_000,
        downloads=10,
    )


def test_a_short_name_is_resolved_before_the_pull(owner, monkeypatch):
    seen = {}

    def fake_resolve(model, quantization, revision):
        seen.update(model=model, quantization=quantization)
        raise ValueError("Model not found: stop here")

    monkeypatch.setattr("hfl.hub.shortname.find", lambda name: _choice(name))
    monkeypatch.setattr("hfl.hub.resolver.resolve", fake_resolve)
    owner.post("/api/pull", json={"model": "llama3.2", "stream": False})
    assert seen == {
        "model": "hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",
        "quantization": "Q4_K_M",
    }


def test_nothing_matching_is_a_404(owner, monkeypatch):
    monkeypatch.setattr("hfl.hub.shortname.find", lambda name: None)
    response = owner.post("/api/pull", json={"model": "nosuchmodel", "stream": False})
    assert response.status_code == 404 and response.json()["code"] == "not_found"


def test_an_unreachable_hub_is_a_503(owner, monkeypatch):
    import httpx

    def offline(name):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr("hfl.hub.shortname.find", offline)
    response = owner.post("/api/pull", json={"model": "llama3.2", "stream": False})
    assert response.status_code == 503 and response.json()["code"] == "hub_unreachable"


def _license():
    return SimpleNamespace(
        license_id="llama3.2", license_name="x", url=None, restrictions=[], gated=False
    )


def test_the_pull_is_registered_under_the_short_name(temp_config, tmp_path):
    from hfl.api.routes_pull import _record_server_pull
    from hfl.models.registry import get_registry

    model = temp_config.models_dir / "m.gguf"
    model.write_bytes(b"GGUF")
    resolved = SimpleNamespace(
        repo_id="unsloth/Llama-3.2-3B-Instruct-GGUF",
        quantization="Q4_K_M",
        revision="main",
        commit_sha="abc",
    )
    _record_server_pull(resolved, Path(model), _license(), "all", alias="llama3.2")
    assert get_registry().get("llama3.2").repo_id == "unsloth/Llama-3.2-3B-Instruct-GGUF"


def test_a_name_already_in_use_is_not_taken(temp_config, tmp_path):
    from hfl.api.routes_pull import _record_server_pull
    from hfl.models.manifest import ModelManifest
    from hfl.models.registry import get_registry

    registry = get_registry()
    registry.add(
        ModelManifest(name="mine", repo_id="a/b", local_path="/x", format="gguf", alias="llama3.2")
    )
    model = temp_config.models_dir / "m.gguf"
    model.write_bytes(b"GGUF")
    resolved = SimpleNamespace(
        repo_id="unsloth/Llama-3.2-3B-Instruct-GGUF",
        quantization="Q4_K_M",
        revision="main",
        commit_sha="abc",
    )
    _record_server_pull(resolved, Path(model), _license(), "all", alias="llama3.2")
    assert registry.get("llama3.2").name == "mine"
    from hfl.models.registry import ModelRegistry

    fresh = ModelRegistry()  # what the pull wrote, read back from disk
    new = [m for m in fresh.list_all() if m.repo_id == "unsloth/Llama-3.2-3B-Instruct-GGUF"]
    assert len(new) == 1 and new[0].alias is None  # registered, without the name


def test_a_tagged_short_name_finds_its_alias(temp_config):
    from hfl.api.model_loader import _canonical_model_name
    from hfl.models.manifest import ModelManifest
    from hfl.models.registry import get_registry

    get_registry().add(
        ModelManifest(
            name="qwen3-8b-q4_k_m",
            repo_id="Qwen/Qwen3-8B-GGUF",
            local_path="/x",
            format="gguf",
            alias="qwen3-8b",
        )
    )
    assert _canonical_model_name("qwen3:8b") == "qwen3-8b-q4_k_m"


def test_an_unknown_tagged_short_name_is_a_404(temp_config):
    from hfl.api.model_loader import _canonical_model_name
    from hfl.exceptions import ModelNotFoundError

    with pytest.raises(ModelNotFoundError):
        _canonical_model_name("qwen3:8b")
