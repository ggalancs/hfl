# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A model its backend cannot load gets an answer that says so, and why —
not "Internal server error" — and MLX falls back to Transformers (found by
the compatibility sweep: distilgpt2 on MLX, "82 parameters not in model";
a DFlash draft GGUF on llama.cpp, "Failed to load model from file")."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


class MLXEngine:  # the name is what the fallback decision reads
    def __init__(self, error: Exception | None = None) -> None:
        self.error, self.is_loaded, self.loads = error, False, 0

    def load(self, path, **kwargs):
        self.loads += 1
        if self.error:
            raise self.error
        self.is_loaded = True

    def generate(self, prompt, config=None):
        return MagicMock(text="ok", tokens_generated=1, tokens_prompt=1)


class TransformersEngine(MLXEngine):
    pass


class LlamaCppEngine(MLXEngine):
    pass


@pytest.fixture
def server(temp_config, monkeypatch):
    from hfl.api.server import app
    from hfl.api.state import reset_state
    from hfl.core.container import get_registry, reset_container
    from hfl.models.manifest import ModelManifest

    reset_container()
    reset_state()
    folder = temp_config.models_dir / "distilgpt2"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(
        '{"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2"}'
    )
    (folder / "model.safetensors").write_bytes(b"x")
    get_registry().add(
        ModelManifest(name="distil", repo_id="d/d", local_path=str(folder), format="safetensors")
    )
    gguf = temp_config.models_dir / "draft.gguf"
    gguf.write_bytes(b"GGUF")
    get_registry().add(
        ModelManifest(name="draft", repo_id="z/d", local_path=str(gguf), format="gguf")
    )
    yield TestClient(app, client=("127.0.0.1", 5555)), temp_config
    reset_state()
    reset_container()


def _generate(client, model: str):
    body = {"model": model, "prompt": "hi", "stream": False, "options": {"num_predict": 1}}
    return client.post("/api/generate", json=body)


def test_mlx_falls_back_to_transformers(server, monkeypatch) -> None:
    client, _ = server
    mlx, tf = (
        MLXEngine(ValueError("Received 82 parameters not in model: h.0.attn")),
        TransformersEngine(),
    )
    monkeypatch.setattr("hfl.api.model_loader.select_engine", lambda path: mlx)
    monkeypatch.setattr("hfl.engine.selector._create_engine", lambda name: tf)
    _generate(client, "distil")
    assert (mlx.loads, tf.loads) == (1, 1) and tf.is_loaded


def test_without_one_the_reason_is_given_without_paths(server, monkeypatch) -> None:
    client, cfg = server
    path_in_error = f"Received 82 parameters not in model {cfg.models_dir}/distilgpt2"
    monkeypatch.setattr(
        "hfl.api.model_loader.select_engine", lambda path: MLXEngine(ValueError(path_in_error))
    )

    def no_transformers(name):
        raise ImportError(name)

    monkeypatch.setattr("hfl.engine.selector._create_engine", no_transformers)
    out = _generate(client, "distil")
    assert out.status_code == 500
    body = out.json()
    assert body["code"] == "ModelLoadError"
    assert "82 parameters not in model" in body["details"] and "MLX" in body["details"]
    assert str(cfg.home_dir) not in out.text and str(cfg.models_dir) not in out.text


def test_a_gguf_llama_cpp_refuses_names_its_architecture(server, monkeypatch) -> None:
    client, cfg = server
    refused = ValueError(f"Failed to load model from file: {cfg.models_dir}/draft.gguf")
    monkeypatch.setattr("hfl.api.model_loader.select_engine", lambda path: LlamaCppEngine(refused))
    monkeypatch.setattr(
        "hfl.converter.gguf_header.read_fields",
        lambda path, keys: {"general.architecture": "dflash2"},
    )
    out = _generate(client, "draft")
    assert out.status_code == 500 and "architecture 'dflash2'" in out.json()["details"]
    assert str(cfg.models_dir) not in out.text


def test_a_gguf_the_bundled_llama_cpp_cannot_load_goes_to_llama_server(server, monkeypatch) -> None:
    client, cfg = server
    bundled = LlamaCppEngine(ValueError("Failed to load model from file: x.gguf"))
    newer = MLXEngine()  # stands in for LlamaServerEngine
    monkeypatch.setattr("hfl.api.model_loader.select_engine", lambda path: bundled)
    monkeypatch.setattr("hfl.engine.llama_server.binary", lambda: "/opt/llama-server")
    asked: list[str] = []

    def create(name):
        asked.append(name)
        return newer

    monkeypatch.setattr("hfl.engine.selector._create_engine", create)
    _generate(client, "draft")
    assert asked == ["llama-server"] and newer.is_loaded
    monkeypatch.setattr("hfl.engine.llama_server.binary", lambda: None)
    asked.clear()
    from hfl.api.state import reset_state

    reset_state()
    out = _generate(client, "draft")
    assert asked == [] and out.json()["code"] == "ModelLoadError"


def test_the_switch_is_logged_as_the_local_audit_recognises_it(server, monkeypatch, caplog) -> None:
    """The audit fails a check whose load went to another engine (the other
    engine answering hid a llama.cpp load that always failed): its pattern
    must match the warning this code writes."""
    import re
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))
    import local_audit

    client, _ = server
    mlx = MLXEngine(ValueError("Received 82 parameters not in model: h.0.attn"))
    monkeypatch.setattr("hfl.api.model_loader.select_engine", lambda path: mlx)
    monkeypatch.setattr("hfl.engine.selector._create_engine", lambda name: TransformersEngine())
    with caplog.at_level("WARNING", logger="hfl.api.model_loader"):
        _generate(client, "distil")
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert re.search(local_audit.ENGINE_SWITCH, logged), logged
