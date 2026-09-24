# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Speech-to-text and image routes: who may download, and the event loop.

Found while adding the OpenAI aliases:

- ``model`` accepted any Hub repo id and the engines called
  ``WhisperModel(model)`` / ``DiffusionPipeline.from_pretrained(model)``,
  which download whatever they are given. Any client, remote ones
  included, could make the server download arbitrary repos — around the
  owner guard and the license check that protect ``/api/pull``. Now only
  the owner may fetch; everyone else gets models already on disk
  (``local_files_only``) or a 404 that says so.
- The loads and the inference ran on the event loop: while an image
  rendered, the whole server — ``/healthz`` included — stopped answering.

Plus the OpenAI paths: ``/v1/audio/transcriptions`` and
``/v1/images/generations``.
"""

from __future__ import annotations

import asyncio
import base64
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.engine.diffusers_engine import ImageResult
from hfl.engine.whisper_engine import WhisperResult as TranscriptionResult
from hfl.engine.whisper_engine import WhisperSegment as TranscriptionSegment

OWNER = ("127.0.0.1", 50000)
REMOTE = ("192.168.1.50", 50000)
AUDIO = {"file": ("a.wav", b"RIFF....WAVE", "audio/wav")}


class FakeWhisper:
    calls: list[dict] = []
    gate: threading.Event | None = None

    def load(self, model, *, local_files_only=False, **_):
        FakeWhisper.calls.append({"model": model, "local_files_only": local_files_only})

    def transcribe(self, audio, language=None, include_segments=False):
        if FakeWhisper.gate is not None:
            FakeWhisper.gate.wait(10)
        segments = [
            TranscriptionSegment(start=0.0, end=1.5, text=" Hello"),
            TranscriptionSegment(start=1.5, end=3.25, text=" world."),
        ]
        return TranscriptionResult(
            text="Hello world.",
            language="en",
            duration_s=3.25,
            segments=segments if include_segments else None,
        )

    def unload(self):
        pass


class FakeDiffusers:
    calls: list[dict] = []

    def load(self, model, *, local_files_only=False, **_):
        FakeDiffusers.calls.append({"model": model, "local_files_only": local_files_only})

    def generate(self, prompt, **kw):
        return ImageResult(
            image_png_base64=base64.b64encode(b"PNG" + prompt.encode()).decode(),
            width=kw["width"],
            height=kw["height"],
            seed=kw.get("seed") or 1,
            duration_s=0.1,
        )

    def unload(self):
        pass


@pytest.fixture(autouse=True)
def fakes(monkeypatch, temp_config):
    from hfl.api import routes_images, routes_transcribe

    FakeWhisper.calls, FakeWhisper.gate = [], None
    FakeDiffusers.calls = []
    monkeypatch.setattr(routes_transcribe, "is_available", lambda: True)
    monkeypatch.setattr(routes_transcribe, "WhisperEngine", FakeWhisper)
    monkeypatch.setattr(routes_transcribe, "whisper_available_locally", lambda m: True)
    monkeypatch.setattr(routes_images, "is_available", lambda: True)
    monkeypatch.setattr(routes_images, "DiffusersEngine", FakeDiffusers)
    monkeypatch.setattr(routes_images, "hub_model_available_locally", lambda m: True)


def _client(peer):
    return TestClient(app, client=peer)


class TestWhoMayDownload:
    @pytest.mark.parametrize(
        ("peer", "headers", "local_only"),
        [
            (OWNER, {}, False),
            (REMOTE, {}, True),
            (OWNER, {"Origin": "https://evil.test"}, True),  # a web page is not the owner
        ],
    )
    def test_transcribe(self, peer, headers, local_only):
        r = _client(peer).post("/api/transcribe", files=AUDIO, headers=headers)
        assert r.status_code == 200
        assert FakeWhisper.calls == [{"model": "small", "local_files_only": local_only}]

    @pytest.mark.parametrize(("peer", "local_only"), [(OWNER, False), (REMOTE, True)])
    def test_images(self, peer, local_only):
        r = _client(peer).post("/api/images/generate", json={"model": "org/sd", "prompt": "a cat"})
        assert r.status_code == 200
        assert FakeDiffusers.calls == [{"model": "org/sd", "local_files_only": local_only}]

    def test_a_remote_client_asking_for_a_missing_model_gets_404(self, monkeypatch):
        from hfl.api import routes_images, routes_transcribe

        monkeypatch.setattr(routes_transcribe, "whisper_available_locally", lambda m: False)
        monkeypatch.setattr(routes_images, "hub_model_available_locally", lambda m: False)
        remote = _client(REMOTE)
        r = remote.post("/api/transcribe", files=AUDIO, data={"model": "org/huge"})
        assert r.status_code == 404 and "owner" in r.text
        r = remote.post("/api/images/generate", json={"model": "org/huge", "prompt": "x"})
        assert r.status_code == 404
        assert FakeWhisper.calls == [] and FakeDiffusers.calls == []

    def test_the_owner_may_fetch_a_missing_model(self, monkeypatch):
        from hfl.api import routes_transcribe

        monkeypatch.setattr(routes_transcribe, "whisper_available_locally", lambda m: False)
        r = _client(OWNER).post("/api/transcribe", files=AUDIO, data={"model": "medium"})
        assert r.status_code == 200
        assert FakeWhisper.calls == [{"model": "medium", "local_files_only": False}]


def test_the_server_keeps_answering_while_a_transcription_runs():
    FakeWhisper.gate = threading.Event()

    async def scenario():
        transport = httpx.ASGITransport(app=app, client=OWNER)
        async with httpx.AsyncClient(transport=transport, base_url="http://hfl") as client:
            started = time.monotonic()
            job = asyncio.create_task(client.post("/api/transcribe", files=AUDIO))
            await asyncio.sleep(0.2)
            try:
                health = await asyncio.wait_for(client.get("/healthz"), timeout=3)
                # Answered while the transcription is still held at the gate:
                # a blocked loop could only answer after the gate gave up (10 s).
                answered_after = time.monotonic() - started
                still_running = not job.done()
            finally:
                FakeWhisper.gate.set()
            done = await asyncio.wait_for(job, timeout=10)
            return health.status_code, done.status_code, still_running, answered_after

    health, done, still_running, answered_after = asyncio.run(scenario())
    assert (health, done, still_running) == (200, 200, True)
    assert answered_after < 3


class TestOpenAITranscriptions:
    def _post(self, **data):
        return _client(OWNER).post("/v1/audio/transcriptions", files=AUDIO, data=data)

    def test_json_is_the_default(self):
        r = self._post(model="whisper-1")
        assert r.json() == {"text": "Hello world."}
        assert FakeWhisper.calls[0]["model"] == "small"  # whisper-1 -> the default size

    def test_a_real_model_name_is_used(self):
        self._post(model="large-v3")
        assert FakeWhisper.calls[0]["model"] == "large-v3"

    def test_text(self):
        r = self._post(model="whisper-1", response_format="text")
        assert r.text == "Hello world." and r.headers["content-type"].startswith("text/plain")

    def test_verbose_json(self):
        body = self._post(model="whisper-1", response_format="verbose_json").json()
        assert body["task"] == "transcribe" and body["language"] == "en"
        assert body["duration"] == 3.25 and body["text"] == "Hello world."
        assert [(s["id"], s["start"], s["end"], s["text"]) for s in body["segments"]] == [
            (0, 0.0, 1.5, " Hello"),
            (1, 1.5, 3.25, " world."),
        ]

    def test_srt(self):
        r = self._post(model="whisper-1", response_format="srt")
        assert r.text == (
            "1\n00:00:00,000 --> 00:00:01,500\nHello\n\n2\n00:00:01,500 --> 00:00:03,250\nworld.\n"
        )

    def test_vtt(self):
        r = self._post(model="whisper-1", response_format="vtt")
        assert r.text.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello\n")

    def test_an_unknown_format_is_400(self):
        assert self._post(model="whisper-1", response_format="docx").status_code == 400

    def test_uploads_beyond_the_json_cap_reach_the_route(self):
        from hfl.api.middleware import RequestBodyLimitMiddleware as M

        assert "/v1/audio/transcriptions" in M.EXCLUDED_PATHS
        assert "/v1/audio/transcriptions-x" not in M.EXCLUDED_PATHS

    def test_a_remote_client_cannot_trigger_a_download_here_either(self, monkeypatch):
        from hfl.api import routes_transcribe

        monkeypatch.setattr(routes_transcribe, "whisper_available_locally", lambda m: False)
        r = _client(REMOTE).post("/v1/audio/transcriptions", files=AUDIO, data={"model": "x/y"})
        assert r.status_code == 404 and FakeWhisper.calls == []


class TestOpenAIImages:
    def test_b64_json(self):
        r = _client(OWNER).post(
            "/v1/images/generations",
            json={"model": "org/sd", "prompt": "a cat", "size": "512x512"},
        )
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["created"], int)
        assert base64.b64decode(body["data"][0]["b64_json"]) == b"PNGa cat"

    def test_n_images(self):
        body = (
            _client(OWNER)
            .post("/v1/images/generations", json={"model": "org/sd", "prompt": "a", "n": 2})
            .json()
        )
        assert len(body["data"]) == 2

    def test_urls_are_not_served(self):
        r = _client(OWNER).post(
            "/v1/images/generations",
            json={"model": "org/sd", "prompt": "a", "response_format": "url"},
        )
        assert r.status_code == 400 and "b64_json" in r.text

    def test_remote_gets_local_only(self):
        _client(REMOTE).post("/v1/images/generations", json={"model": "org/sd", "prompt": "a"})
        assert FakeDiffusers.calls == [{"model": "org/sd", "local_files_only": True}]


def test_the_diffusers_engine_passes_the_flag_to_from_pretrained(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    from hfl.engine import diffusers_engine

    seen: dict = {}

    class Pipeline:
        @classmethod
        def from_pretrained(cls, model, **kw):
            seen.update(kw, model=model)
            return SimpleNamespace(to=lambda device: None)

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    torch.float16, torch.float32 = "f16", "f32"
    diffusers = ModuleType("diffusers")
    diffusers.DiffusionPipeline = Pipeline
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setattr(diffusers_engine, "is_available", lambda: True)

    diffusers_engine.DiffusersEngine().load("org/sd", local_files_only=True)
    assert seen["model"] == "org/sd" and seen["local_files_only"] is True
