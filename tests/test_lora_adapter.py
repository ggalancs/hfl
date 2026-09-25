# SPDX-License-Identifier: Apache-2.0
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

import ctypes
import hashlib
from types import SimpleNamespace
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

        # Adapters must live under the HFL data dir; the loader contains them.
        ad_dir = temp_config.home_dir / "adapters"
        ad_dir.mkdir(parents=True, exist_ok=True)
        ad1, ad2 = ad_dir / "a.gguf", ad_dir / "b.gguf"
        ad1.write_bytes(b"x")
        ad2.write_bytes(b"x")

        manifest = ModelManifest(
            name="withlora",
            repo_id="org/withlora",
            local_path=str(gguf),
            format="gguf",
            adapter_paths=[str(ad1), str(ad2)],
        )
        get_registry().add(manifest)

        ml.load_llm_sync("withlora")
        _, kwargs = fake_engine.load.call_args
        # Loader forwards the contained, resolved adapter paths.
        assert kwargs["lora_paths"] == [str(ad1.resolve()), str(ad2.resolve())]

    def test_the_server_loads_them_too(self, temp_config, monkeypatch):
        """``/api/chat`` on a model created with ADAPTER: the server's load
        passed only the context, so it ran without its adapter (measured)."""
        import asyncio

        from hfl.api import model_loader as ml
        from hfl.api.state import reset_state

        reset_state()
        fake_engine = MagicMock(is_loaded=True, context_size=0)
        monkeypatch.setattr(ml, "select_engine", lambda _p: fake_engine)
        monkeypatch.setattr(ml, "detect_model_type", lambda _p: ml.ModelType.LLM)
        gguf = temp_config.home_dir / "models" / "m.gguf"
        gguf.parent.mkdir(parents=True, exist_ok=True)
        gguf.write_bytes(b"fake")
        adapter = temp_config.home_dir / "adapters" / "a.gguf"
        adapter.parent.mkdir(parents=True, exist_ok=True)
        adapter.write_bytes(b"x")
        get_registry().add(
            ModelManifest(
                name="bard",
                repo_id="org/bard",
                local_path=str(gguf),
                format="gguf",
                adapter_paths=[str(adapter)],
            )
        )
        try:
            asyncio.run(ml.load_llm("bard"))
            _, kwargs = fake_engine.load.call_args
            assert kwargs["lora_paths"] == [str(adapter.resolve())]
        finally:
            reset_state()

    def test_adapter_path_traversal_rejected(self, temp_config, monkeypatch):
        # An ADAPTER path escaping the HFL data dir is refused before load.
        from hfl.api import model_loader as ml

        monkeypatch.setattr(ml, "select_engine", lambda _p: MagicMock())
        monkeypatch.setattr(ml, "detect_model_type", lambda _p: ml.ModelType.LLM)

        gguf = temp_config.home_dir / "models" / "m.gguf"
        gguf.parent.mkdir(parents=True, exist_ok=True)
        gguf.write_bytes(b"fake")

        manifest = ModelManifest(
            name="evil-lora",
            repo_id="org/evil",
            local_path=str(gguf),
            format="gguf",
            adapter_paths=["/etc/passwd"],
        )
        get_registry().add(manifest)

        with pytest.raises(ValueError, match="adapter path rejected"):
            ml.load_llm_sync("evil-lora")

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
    def test_every_adapter_is_applied_after_the_load(
        self,
        temp_config,
        fake_llama,
        monkeypatch,
    ):
        """All of a Modelfile's ADAPTER lines, through the same list as a
        hot-applied adapter: llama.cpp sets a context's adapters as one set,
        so one handed to ``Llama(lora_path=...)`` would be dropped by the
        first hot-apply."""
        gguf = temp_config.home_dir / "m.gguf"
        gguf.write_bytes(b"fake")

        from hfl.engine.llama_cpp import LlamaCppEngine

        applied: list[tuple] = []
        monkeypatch.setattr(
            LlamaCppEngine,
            "apply_lora",
            lambda self, path, scale, adapter_id=None: applied.append((path, scale, adapter_id)),
        )
        engine = LlamaCppEngine()
        engine.load(str(gguf), lora_paths=["/abs/a.gguf", "/abs/b.gguf"])
        assert applied == [("/abs/a.gguf", 1.0, "modelfile-0"), ("/abs/b.gguf", 1.0, "modelfile-1")]
        assert "lora_path" not in fake_llama.instances[-1]

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
    yield TestClient(app, client=("127.0.0.1", 50000))
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


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("llama_cpp") is None,
    reason="llama-cpp-python not installed ([llama] extra)",
)
class TestLlamaCppHotSwap:
    """``apply_lora`` / ``remove_lora`` on llama.cpp's own API, stubbed. The
    real thing was checked on ggml-org/stories15M_MOE and its Shakespeare
    adapter: the same text as llama-server gives with ``--lora``."""

    @pytest.fixture
    def engine(self, monkeypatch):
        from llama_cpp import llama_cpp as lcpp

        from hfl.engine.llama_cpp import LlamaCppEngine

        calls: dict[str, list] = {"set": [], "free": []}
        addresses = {b"/a.gguf": 0x1000, b"/b.gguf": 0x2000}

        def _address(handle):
            return ctypes.cast(handle, ctypes.c_void_p).value

        monkeypatch.setattr(
            lcpp,
            "llama_adapter_lora_init",
            lambda model, path: ctypes.cast(
                ctypes.c_void_p(addresses[path]), lcpp.llama_adapter_lora_p_ctypes
            ),
        )

        def _set(ctx, handles, count, scales):
            calls["set"].append([(_address(handles[i]), round(scales[i], 3)) for i in range(count)])
            return 0

        monkeypatch.setattr(lcpp, "llama_set_adapters_lora", _set)
        monkeypatch.setattr(
            lcpp, "llama_adapter_lora_free", lambda h: calls["free"].append(_address(h))
        )
        engine = LlamaCppEngine()
        engine._model = SimpleNamespace(
            model=1, ctx=2, reset=lambda: calls.setdefault("reset", []).append(1)
        )
        return engine, calls

    def test_adapters_are_set_as_one_list(self, engine):
        engine, calls = engine
        engine.apply_lora("/a.gguf", 0.5, adapter_id="a")
        engine.apply_lora("/b.gguf", 1.0, adapter_id="b")
        assert calls["set"][-1] == [(0x1000, 0.5), (0x2000, 1.0)]
        engine.remove_lora("a")
        assert calls["set"][-1] == [(0x2000, 1.0)]
        assert calls["free"] == [0x1000]
        # The cached prompt was computed with other weights: forgotten.
        assert len(calls["reset"]) == 3

    def test_a_file_llama_cpp_cannot_load_is_refused(self, engine, monkeypatch):
        from llama_cpp import llama_cpp as lcpp

        engine, calls = engine
        monkeypatch.setattr(lcpp, "llama_adapter_lora_init", lambda model, path: None)
        with pytest.raises(ValueError, match="not a LoRA adapter"):
            engine.apply_lora("/base-model.gguf", 1.0, adapter_id="x")
        assert engine._loras == [] and calls["set"] == []

    def test_an_unknown_id_is_an_error(self, engine):
        engine, _ = engine
        with pytest.raises(RuntimeError, match="not applied"):
            engine.remove_lora("nope")
