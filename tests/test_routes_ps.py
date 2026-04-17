# SPDX-License-Identifier: HRUL-1.0
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
        assert body == {"models": []}


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

        # Expiry: no keep_alive deadline set ⇒ null
        assert entry["expires_at"] is None

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
