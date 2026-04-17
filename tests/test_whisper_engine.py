# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the Whisper engine + ``/api/transcribe`` route (Phase 16)."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hfl.engine import whisper_engine
from hfl.engine.whisper_engine import WhisperEngine, is_available


class TestAvailability:
    def test_false_when_no_backend(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)
        monkeypatch.delitem(sys.modules, "whisper", raising=False)

        real_import = __import__

        def _deny(name, *a, **k):
            if name in ("faster_whisper", "whisper"):
                raise ImportError(name)
            return real_import(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", _deny)
        assert is_available() is False

    def test_true_when_faster_whisper_importable(self, monkeypatch):
        fake = ModuleType("faster_whisper")
        monkeypatch.setitem(sys.modules, "faster_whisper", fake)
        assert is_available() is True


class TestLoad:
    def test_load_raises_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(whisper_engine, "is_available", lambda: False)
        engine = WhisperEngine()
        with pytest.raises(RuntimeError):
            engine.load("small")

    def test_load_uses_faster_whisper_when_present(self, monkeypatch):
        monkeypatch.setattr(whisper_engine, "is_available", lambda: True)

        class _FakeModel:
            def __init__(self, model, device="auto", compute_type="auto"):
                self.model = model
                self.device = device
                self.compute_type = compute_type

            def transcribe(self, audio, language=None, beam_size=1):
                return iter([]), SimpleNamespace(language="en")

        fake_module = ModuleType("faster_whisper")
        fake_module.WhisperModel = _FakeModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

        engine = WhisperEngine()
        engine.load("small")
        assert engine.backend == "faster_whisper"
        assert engine.is_loaded


class TestTranscribe:
    def test_transcribe_roundtrip_faster_whisper(self, monkeypatch):
        monkeypatch.setattr(whisper_engine, "is_available", lambda: True)

        class _Segment:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text

        class _FakeModel:
            def __init__(self, *_a, **_k):
                pass

            def transcribe(self, audio, language=None, beam_size=1):
                segs = [_Segment(0.0, 1.0, "hello "), _Segment(1.0, 2.0, "world")]
                return iter(segs), SimpleNamespace(language="en")

        fake_module = ModuleType("faster_whisper")
        fake_module.WhisperModel = _FakeModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

        engine = WhisperEngine()
        engine.load("small")
        result = engine.transcribe(b"fake_audio", include_segments=True)
        assert result.text == "hello world"
        assert result.language == "en"
        assert result.segments is not None
        assert len(result.segments) == 2

    def test_transcribe_raises_when_not_loaded(self):
        engine = WhisperEngine()
        with pytest.raises(RuntimeError):
            engine.transcribe(b"x")


# ----------------------------------------------------------------------
# /api/transcribe route
# ----------------------------------------------------------------------


class TestTranscribeRoute:
    def test_501_when_no_backend(self, monkeypatch, temp_config):
        from hfl.api import routes_transcribe

        monkeypatch.setattr(routes_transcribe, "is_available", lambda: False)
        from hfl.api.server import app

        client = TestClient(app)
        resp = client.post(
            "/api/transcribe",
            data={"model": "small"},
            files={"file": ("x.wav", b"fakebytes", "audio/wav")},
        )
        assert resp.status_code == 501

    def test_happy_path(self, monkeypatch, temp_config):
        from hfl.api import routes_transcribe

        monkeypatch.setattr(routes_transcribe, "is_available", lambda: True)

        class _FakeEngine:
            def __init__(self):
                self._loaded = False

            def load(self, model, device=None, compute_type=None):
                self._loaded = True

            def unload(self):
                self._loaded = False

            @property
            def is_loaded(self):
                return self._loaded

            def transcribe(self, audio, language=None, include_segments=False):
                return whisper_engine.WhisperResult(
                    text="hola",
                    language="es",
                    duration_s=0.1,
                )

        monkeypatch.setattr(routes_transcribe, "WhisperEngine", lambda: _FakeEngine())
        from hfl.api.server import app

        client = TestClient(app)
        resp = client.post(
            "/api/transcribe",
            data={"model": "small", "language": "es"},
            files={"file": ("x.wav", b"fakebytes", "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["text"] == "hola"
        assert body["language"] == "es"

    def test_413_on_oversized_audio(self, monkeypatch, temp_config):
        from hfl.api import routes_transcribe

        monkeypatch.setattr(routes_transcribe, "is_available", lambda: True)
        monkeypatch.setattr(routes_transcribe, "_MAX_AUDIO_BYTES", 16)
        from hfl.api.server import app

        client = TestClient(app)
        resp = client.post(
            "/api/transcribe",
            data={"model": "small"},
            files={"file": ("x.wav", b"A" * 32, "audio/wav")},
        )
        assert resp.status_code == 413
