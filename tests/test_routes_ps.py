# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the Ollama-compatible ``GET /api/ps`` endpoint.

Pins the wire-format exactly — Open WebUI, ollama-python and LangChain
tooling key off these field names.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.routes_ps import _manifest_digest, _size_vram_estimate
from hfl.api.server import app
from hfl.api.state import get_state, reset_state


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


@pytest.fixture
def owner(temp_config):
    """A loopback peer: the server's owner, who sees the memory summary."""
    reset_state()
    yield TestClient(app, client=("127.0.0.1", 50000))
    reset_state()


@pytest.fixture
def llm_manifest():
    """A typical LLM manifest — qwen-style."""
    from hfl.models.manifest import ModelManifest

    return ModelManifest(
        name="qwen-coder:7b",
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        local_path="/tmp/qwen-coder-7b.gguf",
        format="gguf",
        architecture="qwen",
        parameters="7B",
        quantization="Q4_K_M",
        size_bytes=4_200_000_000,
        file_hash="sha256:abc123def456" + "0" * 52,
    )


@pytest.fixture
def tts_manifest():
    from hfl.models.manifest import ModelManifest

    return ModelManifest(
        name="bark-small",
        repo_id="suno/bark-small",
        local_path="/tmp/bark",
        format="safetensors",
        architecture="bark",
        parameters="100M",
        size_bytes=400_000_000,
    )


class TestRoutesPsEmpty:
    def test_empty_pool_returns_empty_list(self, client):
        """No model loaded → ``{"models": []}``."""
        response = client.get("/api/ps")
        assert response.status_code == 200
        body = response.json()
        assert body["models"] == []
        # The only other key is HFL's memory summary (ignored by Ollama clients).
        assert set(body) <= {"models", "memory"}


class TestRoutesPsSingleLLM:
    def test_llm_only_emits_one_entry(self, client, llm_manifest):
        """A single loaded LLM surfaces with the Ollama shape."""
        state = get_state()
        state.engine = MagicMock()
        state.engine.is_loaded = True
        state.current_model = llm_manifest

        response = client.get("/api/ps")
        assert response.status_code == 200
        entries = response.json()["models"]
        assert len(entries) == 1

        entry = entries[0]
        # Every field the Ollama contract promises
        assert set(entry.keys()) >= {
            "name",
            "model",
            "size",
            "digest",
            "details",
            "expires_at",
            "size_vram",
        }
        assert entry["name"] == "qwen-coder:7b"
        assert entry["model"] == "qwen-coder:7b"
        assert entry["size"] == 4_200_000_000

        # Digest carries the sha256 prefix required by Ollama tooling.
        assert entry["digest"].startswith("sha256:") or entry["digest"].startswith("sha")

        # Details sub-object
        assert entry["details"]["format"] == "gguf"
        assert entry["details"]["family"] == "qwen"
        assert entry["details"]["parameter_size"] == "7B"
        assert entry["details"]["quantization_level"] == "Q4_K_M"

        # Expiry: a loaded model follows the default keep_alive (5m) from
        # load time, as in Ollama.
        expires = datetime.strptime(entry["expires_at"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
        remaining = (expires - datetime.now(timezone.utc)).total_seconds()
        assert 240 < remaining <= 300

    def test_never_expiring_model_reports_null(self, client, llm_manifest):
        state = get_state()
        state.engine = MagicMock(is_loaded=True)
        state.current_model = llm_manifest
        state.set_keep_alive(llm_manifest.name, None)  # keep_alive=-1
        assert client.get("/api/ps").json()["models"][0]["expires_at"] is None

    def test_digest_falls_back_to_identity_hash_when_no_file_hash(self, client, llm_manifest):
        """A manifest without ``file_hash`` still gets a deterministic digest."""
        llm_manifest.file_hash = None
        state = get_state()
        state.engine = MagicMock(is_loaded=True)
        state.current_model = llm_manifest

        response = client.get("/api/ps")
        entry = response.json()["models"][0]
        assert entry["digest"].startswith("sha256:")
        assert len(entry["digest"]) == 7 + 64  # "sha256:" + 64 hex chars


class TestRoutesPsWithTTS:
    def test_llm_and_tts_both_listed(self, client, llm_manifest, tts_manifest):
        """LLM + TTS both loaded → two entries (LLM first)."""
        state = get_state()
        state.engine = MagicMock(is_loaded=True)
        state.current_model = llm_manifest
        state.tts_engine = MagicMock(is_loaded=True)
        state.current_tts_model = tts_manifest

        response = client.get("/api/ps")
        entries = response.json()["models"]
        assert len(entries) == 2
        names = [e["name"] for e in entries]
        assert names == ["qwen-coder:7b", "bark-small"]


class TestRoutesPsSizeVram:
    def test_engine_reports_zero_vram_when_on_cpu(self, client, llm_manifest):
        """An engine explicitly signalling CPU-only reports size_vram=0."""
        state = get_state()
        engine = MagicMock(is_loaded=True)
        engine.memory_used_bytes = MagicMock(return_value=0)
        state.engine = engine
        state.current_model = llm_manifest

        entry = client.get("/api/ps").json()["models"][0]
        assert entry["size_vram"] == 0

    def test_engine_vram_report_wins_over_manifest(self, client, llm_manifest):
        """If the engine reports VRAM, /api/ps uses that number."""
        state = get_state()
        engine = MagicMock(is_loaded=True)
        engine.memory_used_bytes = MagicMock(return_value=3_000_000_000)
        state.engine = engine
        state.current_model = llm_manifest

        entry = client.get("/api/ps").json()["models"][0]
        assert entry["size_vram"] == 3_000_000_000

    def test_fallback_to_manifest_size_when_engine_silent(self, client, llm_manifest):
        """No memory_used_bytes → conservative upper bound = file size."""
        state = get_state()
        state.engine = MagicMock(is_loaded=True, spec=[])  # spec=[] strips attrs
        state.current_model = llm_manifest

        entry = client.get("/api/ps").json()["models"][0]
        assert entry["size_vram"] == llm_manifest.size_bytes


class TestRoutesPsKeepAliveDeadline:
    def test_expires_at_reflects_keep_alive_deadline(self, client, llm_manifest):
        """When a keep_alive deadline is registered, /api/ps emits it."""
        state = get_state()
        state.engine = MagicMock(is_loaded=True)
        state.current_model = llm_manifest

        deadline = datetime(2026, 4, 17, 15, 30, 0, tzinfo=timezone.utc)
        state.set_keep_alive_deadline(llm_manifest.name, deadline)

        entry = client.get("/api/ps").json()["models"][0]
        # ISO-8601 with trailing Z — Ollama convention
        assert entry["expires_at"] is not None
        assert entry["expires_at"].startswith("2026-04-17T15:30:00")
        assert entry["expires_at"].endswith("Z")

    def test_clearing_deadline_restores_null(self, client, llm_manifest):
        """set_keep_alive_deadline(None) clears the field."""
        state = get_state()
        state.engine = MagicMock(is_loaded=True)
        state.current_model = llm_manifest
        state.set_keep_alive_deadline(llm_manifest.name, datetime(2026, 1, 1, tzinfo=timezone.utc))
        state.set_keep_alive_deadline(llm_manifest.name, None)

        entry = client.get("/api/ps").json()["models"][0]
        assert entry["expires_at"] is None


class TestRoutesPsHelpers:
    def test_manifest_digest_deterministic(self, llm_manifest):
        """Two calls for the same manifest produce the identical digest."""
        llm_manifest.file_hash = None
        d1 = _manifest_digest(llm_manifest)
        d2 = _manifest_digest(llm_manifest)
        assert d1 == d2

    def test_manifest_digest_changes_with_identity(self, llm_manifest):
        """Changing name / path changes the identity digest."""
        llm_manifest.file_hash = None
        d1 = _manifest_digest(llm_manifest)
        llm_manifest.local_path = "/other/path.gguf"
        d2 = _manifest_digest(llm_manifest)
        assert d1 != d2

    def test_size_vram_handles_engine_raising(self, llm_manifest):
        """Engine's memory_used_bytes raising → fall back to manifest."""
        engine = MagicMock(is_loaded=True)
        engine.memory_used_bytes = MagicMock(side_effect=RuntimeError("broken probe"))
        got = _size_vram_estimate(llm_manifest, engine)
        assert got == llm_manifest.size_bytes


class TestRoutesPsListsEveryResident:
    """Several models can be resident at once; ``/api/ps`` lists each one,
    most recently used first, with its memory footprint, plus the
    machine's memory against the budget."""

    @staticmethod
    def _register(manifest, footprint=0):
        import asyncio

        state = get_state()
        engine = MagicMock(is_loaded=True)
        asyncio.run(state.set_llm_engine(engine, manifest))
        if footprint:
            state.resident(manifest.name).footprint = footprint
        return engine

    def test_two_residents_are_both_listed(self, client, llm_manifest, tts_manifest):
        from hfl.models.manifest import ModelManifest

        other = ModelManifest(
            name="llama:8b", repo_id="x/y", local_path="/tmp/l.gguf", format="gguf", size_bytes=5
        )
        self._register(llm_manifest)
        self._register(other)

        names = [m["name"] for m in client.get("/api/ps").json()["models"]]
        assert names == ["llama:8b", "qwen-coder:7b"]

    def test_size_is_the_footprint_when_known(self, client, llm_manifest):
        self._register(llm_manifest, footprint=9_000_000_000)
        entry = client.get("/api/ps").json()["models"][0]
        assert entry["size"] == 9_000_000_000

    def test_no_double_count_with_the_pointer(self, client, llm_manifest):
        self._register(llm_manifest)
        names = [m["name"] for m in client.get("/api/ps").json()["models"]]
        assert names.count(llm_manifest.name) == 1

    def test_memory_summary(self, owner):
        body = owner.get("/api/ps").json()
        memory = body.get("memory")
        pytest.importorskip("psutil")
        assert memory is not None
        assert memory["total_bytes"] > 0
        assert 0 < memory["budget_percent"] <= 100
        assert memory["budget_bytes"] == int(memory["total_bytes"] * memory["budget_percent"] / 100)


def test_memory_summary_includes_a_measured_gpu(owner, monkeypatch):
    from hfl.engine.residency import MemoryView

    pytest.importorskip("psutil")
    gib = 1024**3
    monkeypatch.setattr(
        "hfl.engine.residency.current_gpu_memory",
        lambda: MemoryView(total=24 * gib, in_use=6 * gib, hfl_rss=5 * gib),
    )
    gpu = owner.get("/api/ps").json()["memory"]["gpu"]
    assert gpu["total_bytes"] == 24 * gib and gpu["hfl_bytes"] == 5 * gib
    assert gpu["in_use_percent"] == 25.0


def test_no_gpu_key_without_a_discrete_gpu(owner, monkeypatch):
    pytest.importorskip("psutil")
    monkeypatch.setattr("hfl.engine.residency.current_gpu_memory", lambda: None)
    assert "gpu" not in owner.get("/api/ps").json()["memory"]


def test_a_remote_client_sees_no_host_memory(client):
    """Host RAM and VRAM are the owner's to see, like any admin view."""
    body = client.get("/api/ps").json()
    assert "memory" not in body
    assert body["models"] == []
