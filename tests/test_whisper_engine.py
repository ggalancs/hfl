# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the Whisper engine + ``/api/transcribe`` route (Phase 16)."""

from __future__ import annotations

import io
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
            def __init__(self, model, device="auto", compute_type="auto", local_files_only=False):
                self.model = model
                self.device = device
                self.compute_type = compute_type
                self.local_files_only = local_files_only

            def transcribe(self, audio, language=None, beam_size=1):
                return iter([]), SimpleNamespace(language="en")

        fake_module = ModuleType("faster_whisper")
        fake_module.WhisperModel = _FakeModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

        engine = WhisperEngine()
        engine.load("small")
        assert engine.backend == "faster_whisper"
        assert engine.is_loaded
        assert engine._model.local_files_only is False
        engine.load("small", local_files_only=True)
        assert engine._model.local_files_only is True  # reaches the backend


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
                received.append(audio)
                segs = [_Segment(0.0, 1.0, "hello "), _Segment(1.0, 2.0, "world")]
                return iter(segs), SimpleNamespace(language="en")

        fake_module = ModuleType("faster_whisper")
        fake_module.WhisperModel = _FakeModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
        received: list = []
        samples = object()
        decoded: list[bytes] = []
        monkeypatch.setattr(
            whisper_engine, "_decode_audio", lambda audio: decoded.append(audio) or samples
        )

        engine = WhisperEngine()
        engine.load("small")
        result = engine.transcribe(b"fake_audio", include_segments=True)
        # HFL decodes; faster-whisper gets the samples, never the bytes.
        assert decoded == [b"fake_audio"] and received == [samples]
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

        client = TestClient(app, client=("127.0.0.1", 50000))  # the owner
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

            def load(self, model, device=None, compute_type=None, local_files_only=False):
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

        client = TestClient(app, client=("127.0.0.1", 50000))  # the owner
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

        client = TestClient(app, client=("127.0.0.1", 50000))  # the owner
        resp = client.post(
            "/api/transcribe",
            data={"model": "small"},
            files={"file": ("x.wav", b"A" * 32, "audio/wav")},
        )
        assert resp.status_code == 413


def _wav(rate: int = 44100, channels: int = 2, seconds: float = 0.5) -> bytes:
    """A 440 Hz tone at half scale, 16-bit PCM."""
    import math
    import struct
    import wave

    frames = bytearray()
    for i in range(int(rate * seconds)):
        value = int(16383 * math.sin(2 * math.pi * 440 * i / rate))
        frames += struct.pack("<h", value) * channels
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return out.getvalue()


class TestDecodeAudio:
    """HFL's own decoder (PyAV), which lets [stt] use PyAV 19: faster-whisper
    1.2.1's decoder passes av.open an argument PyAV 19 dropped."""

    def test_mono_16k_float32(self):
        pytest.importorskip("av")
        import numpy as np

        samples = whisper_engine._decode_audio(_wav())
        assert samples.dtype == np.float32 and samples.ndim == 1
        assert abs(len(samples) - 8000) <= 80  # 0.5 s at 16 kHz
        assert 0.4 < float(np.abs(samples).max()) < 0.6  # the tone at half scale

    def test_the_same_samples_as_faster_whispers_decoder(self):
        av = pytest.importorskip("av")
        audio_module = pytest.importorskip("faster_whisper.audio")
        import numpy as np

        if int(av.__version__.split(".")[0]) >= 19:
            pytest.skip("faster-whisper's own decoder does not run on PyAV 19")
        ours = whisper_engine._decode_audio(_wav())
        theirs = audio_module.decode_audio(io.BytesIO(_wav()))
        assert np.array_equal(ours, theirs)

    def test_not_audio_raises_as_before(self):
        av = pytest.importorskip("av")
        with pytest.raises(av.error.FFmpegError):
            whisper_engine._decode_audio(b"this is not audio")
