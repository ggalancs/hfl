# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for LoRA adapter plumbing (Phase 8 P3-2 part 2).

Real LoRA load exercises llama-cpp-python and a gigabyte-sized
adapter, which is out of scope for the test suite. Instead we
verify the wiring:

- ``manifest.adapter_paths`` flows into ``engine.load`` as
  ``lora_paths=[...]``.
- ``LlamaCppEngine.load`` hands the first path to llama-cpp's
  ``lora_path`` kwarg and warns on the rest.
- ``POST /api/create`` with a Modelfile containing ``ADAPTER``
  persists the path on the manifest.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import reset_state
from hfl.models.manifest import ModelManifest
from hfl.models.registry import get_registry, reset_registry

# ----------------------------------------------------------------------
# Model-loader plumbing
# ----------------------------------------------------------------------


class TestModelLoaderPassesAdapterPaths:
    def test_adapter_paths_reach_engine_load(self, temp_config, monkeypatch):
        # Fake the engine factory so we don't instantiate a real Llama.
        from hfl.api import model_loader as ml

        fake_engine = MagicMock()

        def fake_select_engine(path):  # noqa: ANN001
            return fake_engine

        monkeypatch.setattr(ml, "select_engine", fake_select_engine)
        monkeypatch.setattr(
            ml,
            "detect_model_type",
            lambda _p: ml.ModelType.LLM,
        )

        gguf = temp_config.home_dir / "models" / "m.gguf"
        gguf.parent.mkdir(parents=True, exist_ok=True)
        gguf.write_bytes(b"fake")

        manifest = ModelManifest(
            name="withlora",
            repo_id="org/withlora",
            local_path=str(gguf),
            format="gguf",
            adapter_paths=["/abs/a.gguf", "/abs/b.gguf"],
        )
        get_registry().add(manifest)

        ml.load_llm_sync("withlora")
        _, kwargs = fake_engine.load.call_args
        assert kwargs["lora_paths"] == ["/abs/a.gguf", "/abs/b.gguf"]

    def test_no_adapters_no_lora_kwarg(self, temp_config, monkeypatch):
        from hfl.api import model_loader as ml

        fake_engine = MagicMock()
        monkeypatch.setattr(ml, "select_engine", lambda _p: fake_engine)
        monkeypatch.setattr(
            ml,
            "detect_model_type",
            lambda _p: ml.ModelType.LLM,
        )

        gguf = temp_config.home_dir / "models" / "m.gguf"
        gguf.parent.mkdir(parents=True, exist_ok=True)
        gguf.write_bytes(b"fake")

        manifest = ModelManifest(
            name="plain",
            repo_id="org/plain",
            local_path=str(gguf),
            format="gguf",
        )
        get_registry().add(manifest)

        ml.load_llm_sync("plain")
        _, kwargs = fake_engine.load.call_args
        assert "lora_paths" not in kwargs


# ----------------------------------------------------------------------
# LlamaCppEngine.load → Llama(lora_path=...)
# ----------------------------------------------------------------------


class _FakeLlama:
    """Captures the kwargs Llama would have been instantiated with."""

    instances: list[dict] = []

    def __init__(self, **kwargs):  # noqa: D401
        _FakeLlama.instances.append(kwargs)
        self.kwargs = kwargs
        self.chat_handlers = None

    def tokenize(self, *_a, **_k):
        return [0, 1, 2]


@pytest.fixture
def fake_llama(monkeypatch):
    import hfl.engine.llama_cpp as lc

    _FakeLlama.instances.clear()
    monkeypatch.setattr(lc, "Llama", _FakeLlama)
    # Skip the GGUF header probe — we're using a fake file.
    monkeypatch.setattr(lc, "_read_gguf_model_info", lambda _p: None)
    return _FakeLlama


class TestLlamaCppEngineLoraWiring:
    def test_first_lora_path_maps_to_lora_path_kwarg(
        self,
        temp_config,
        fake_llama,
        monkeypatch,
    ):
        gguf = temp_config.home_dir / "m.gguf"
        gguf.write_bytes(b"fake")

        from hfl.engine.llama_cpp import LlamaCppEngine

        engine = LlamaCppEngine()
        engine.load(str(gguf), lora_paths=["/abs/a.gguf"])
        assert fake_llama.instances[-1].get("lora_path") == "/abs/a.gguf"

    def test_only_first_adapter_is_used(
        self,
        temp_config,
        fake_llama,
        monkeypatch,
    ):
        gguf = temp_config.home_dir / "m.gguf"
        gguf.write_bytes(b"fake")

        from hfl.engine import llama_cpp as lc
        from hfl.engine.llama_cpp import LlamaCppEngine

        warnings: list[str] = []
        monkeypatch.setattr(
            lc.logger,
            "warning",
            lambda msg, *args, **kwargs: warnings.append(msg % args if args else msg),
        )

        engine = LlamaCppEngine()
        engine.load(str(gguf), lora_paths=["/abs/a.gguf", "/abs/b.gguf"])
        assert fake_llama.instances[-1].get("lora_path") == "/abs/a.gguf"
        assert any("additional LoRA" in w for w in warnings)

    def test_no_lora_paths_means_no_lora_path_kwarg(
        self,
        temp_config,
        fake_llama,
    ):
        gguf = temp_config.home_dir / "m.gguf"
        gguf.write_bytes(b"fake")

        from hfl.engine.llama_cpp import LlamaCppEngine

        engine = LlamaCppEngine()
        engine.load(str(gguf))
        assert "lora_path" not in fake_llama.instances[-1]


# ----------------------------------------------------------------------
# /api/create persists ADAPTER on the manifest
# ----------------------------------------------------------------------


@pytest.fixture
def client(temp_config):
    reset_state()
    reset_registry()
    yield TestClient(app)
    reset_state()


def _parent(temp_config):
    gguf = temp_config.home_dir / "models" / "parent.gguf"
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(b"fake")
    m = ModelManifest(
        name="parent",
        repo_id="org/parent",
        local_path=str(gguf),
        format="gguf",
        size_bytes=gguf.stat().st_size,
        file_hash=hashlib.sha256(gguf.read_bytes()).hexdigest(),
    )
    get_registry().add(m)


class TestCreateRoutePersistsAdapter:
    def test_adapter_lines_land_on_manifest(self, client, temp_config):
        _parent(temp_config)
        body = "FROM parent\nADAPTER /abs/lora-a.gguf\nADAPTER /abs/lora-b.gguf\n"
        resp = client.post(
            "/api/create",
            json={"model": "tuned", "modelfile": body, "stream": False},
        )
        assert resp.status_code == 200
        derived = get_registry().get("tuned")
        assert derived is not None
        assert derived.adapter_paths == ["/abs/lora-a.gguf", "/abs/lora-b.gguf"]
