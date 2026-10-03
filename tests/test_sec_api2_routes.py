# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Security audit, group api2: diagnostics, host details, Hub calls, media.

Each class pins one finding:

- verify / benchmark generated on the model with no owner guard and
  around the dispatcher (verify on the event loop itself: the whole
  server froze); benchmark's ``prompt_lengths`` had no bound.
- /api/show's Modelfile and /api/lora handed host paths to anyone.
- /api/recommend and /api/compliance/dashboard handed host hardware and
  the owner's Hub credential status to anyone.
- /api/discover?refresh, /api/recommend and /api/draft/recommend let
  anyone spend Hub calls on the owner's token.
- image and speech routes took filesystem paths as the model (an
  existence oracle), ran around the dispatcher, and took 99999x99999.
"""

from __future__ import annotations

import asyncio
import base64
import threading
import time
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state
from hfl.engine.diffusers_engine import ImageResult
from hfl.engine.whisper_engine import WhisperResult

OWNER = ("127.0.0.1", 50000)
REMOTE = ("203.0.113.9", 5000)
SECRET_DIR = "/Users/secret-owner/private-models"


def _client(peer):
    return TestClient(app, client=peer)


def _accepted() -> int:
    from hfl.core import get_dispatcher

    return get_dispatcher().snapshot().accepted_total


@pytest.fixture
def manifest():
    from hfl.models.manifest import ModelManifest

    return ModelManifest(
        name="qwen-coder-7b",
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        local_path=f"{SECRET_DIR}/qwen-coder-7b.gguf",
        format="gguf",
        architecture="qwen",
        parameters="7B",
        adapter_paths=[f"{SECRET_DIR}/lora/style.gguf"],
    )


@pytest.fixture
def engine(temp_config, manifest):
    reset_state()
    state = get_state()
    eng = MagicMock(
        spec=["generate", "is_loaded", "supports_concurrent_inference", "parallel_slots"]
    )
    eng.is_loaded = True
    result = MagicMock()
    result.text = "ok"
    result.tokens_generated = 1
    eng.generate = MagicMock(return_value=result)
    state.engine = eng
    state.current_model = manifest
    yield eng
    reset_state()


# ---------------------------------------------------------------------------
# 1 + 2. verify / benchmark: owner-only, dispatched, bounded
# ---------------------------------------------------------------------------


class TestDiagnosticsAreTheOwners:
    @pytest.mark.parametrize(
        ("path", "body"),
        [
            ("/api/verify/qwen-coder-7b", None),
            ("/api/benchmark/qwen-coder-7b", {"prompt_lengths": [16], "runs_per_length": 1}),
        ],
    )
    def test_a_remote_peer_is_refused_before_any_generation(self, engine, path, body):
        r = _client(REMOTE).post(path, json=body)
        assert r.status_code == 403
        engine.generate.assert_not_called()

    def test_verify_goes_through_the_dispatcher(self, engine):
        before = _accepted()
        r = _client(OWNER).post("/api/verify/qwen-coder-7b")
        assert r.status_code == 200, r.text
        engine.generate.assert_called()
        assert _accepted() == before + 1

    def test_benchmark_goes_through_the_dispatcher(self, engine):
        before = _accepted()
        r = _client(OWNER).post(
            "/api/benchmark/qwen-coder-7b",
            json={"prompt_lengths": [16, 256], "runs_per_length": 2, "stream": False},
        )
        assert r.status_code == 200, r.text
        assert len(r.json()["summaries"]) == 2
        assert _accepted() == before + 4  # one slot per measurement

    def test_verify_waits_on_its_own_models_queue(self, engine):
        # A concurrent engine (llama-server) has its own dispatcher: verify
        # must take a slot there, not on the global one beside it.
        from hfl.core import dispatcher_for

        engine.supports_concurrent_inference = True
        engine.parallel_slots = 2
        own = dispatcher_for(engine)
        before_own, before_global = own.snapshot().accepted_total, _accepted()
        r = _client(OWNER).post("/api/verify/qwen-coder-7b")
        assert r.status_code == 200, r.text
        assert own.snapshot().accepted_total == before_own + 1
        assert _accepted() == before_global

    def test_benchmark_waits_on_its_own_models_queue(self, engine):
        from hfl.core import dispatcher_for

        engine.supports_concurrent_inference = True
        engine.parallel_slots = 2
        own = dispatcher_for(engine)
        before_own, before_global = own.snapshot().accepted_total, _accepted()
        r = _client(OWNER).post(
            "/api/benchmark/qwen-coder-7b",
            json={"prompt_lengths": [16], "runs_per_length": 1, "stream": False},
        )
        assert r.status_code == 200, r.text
        assert own.snapshot().accepted_total == before_own + 1
        assert _accepted() == before_global

    def test_benchmark_runs_are_not_counted_as_served_generations(self, engine):
        from hfl.metrics import get_metrics

        before = get_metrics().tokens_generated
        r = _client(OWNER).post(
            "/api/benchmark/qwen-coder-7b",
            json={"prompt_lengths": [16], "runs_per_length": 2, "stream": False},
        )
        assert r.status_code == 200, r.text
        assert get_metrics().tokens_generated == before

    @pytest.mark.parametrize(
        "body",
        [
            {"prompt_lengths": [16] * 1000},
            {"prompt_lengths": []},
            {"prompt_lengths": [0]},
            {"runs_per_length": 21},
            {"max_tokens": 2049},
        ],
    )
    def test_benchmark_sizes_are_bounded(self, engine, body):
        r = _client(OWNER).post("/api/benchmark/qwen-coder-7b", json=body)
        assert r.status_code == 422
        engine.generate.assert_not_called()

    def test_the_server_answers_while_verify_generates(self, engine):
        gate = threading.Event()
        result = MagicMock()
        result.text = "ok"

        def slow_generate(*_a, **_k):
            gate.wait(10)
            return result

        engine.generate.side_effect = slow_generate

        async def scenario():
            transport = httpx.ASGITransport(app=app, client=OWNER)
            async with httpx.AsyncClient(transport=transport, base_url="http://hfl") as client:
                job = asyncio.create_task(client.post("/api/verify/qwen-coder-7b"))
                await asyncio.sleep(0.2)
                try:
                    started = time.monotonic()
                    health = await asyncio.wait_for(client.get("/healthz"), timeout=3)
                    waited = time.monotonic() - started
                    still_running = not job.done()
                finally:
                    gate.set()
                verify = await job
                return health, waited, still_running, verify

        health, waited, still_running, verify = asyncio.run(scenario())
        assert health.status_code == 200
        assert still_running and waited < 3
        assert verify.status_code == 200


# ---------------------------------------------------------------------------
# 3. host paths in /api/show and /api/lora
# ---------------------------------------------------------------------------


class TestHostPathsAreTheOwners:
    @pytest.fixture
    def registered(self, temp_config, manifest):
        from hfl.models.registry import get_registry

        get_registry().add(manifest)
        return manifest

    def test_show_hides_paths_from_a_remote_peer(self, registered):
        r = _client(REMOTE).post("/api/show", json={"model": registered.name})
        assert r.status_code == 200
        modelfile = r.json()["modelfile"]
        assert SECRET_DIR not in r.text
        assert f"FROM {registered.name}" in modelfile
        assert "ADAPTER style.gguf" in modelfile

    def test_show_keeps_paths_for_the_owner(self, registered):
        r = _client(OWNER).post("/api/show", json={"model": registered.name})
        assert f"FROM {registered.local_path}" in r.json()["modelfile"]

    @pytest.fixture
    def adapter(self):
        from hfl.engine.lora import AdapterInfo, get_registry, reset_registry

        reset_registry()
        get_registry().add(
            AdapterInfo(
                adapter_id="a1",
                path=f"{SECRET_DIR}/lora/style.gguf",
                name="style",
                scale=1.0,
                engine_id="e",
            )
        )
        yield
        reset_registry()

    def test_lora_list_hides_paths_from_a_remote_peer(self, temp_config, adapter):
        r = _client(REMOTE).get("/api/lora")
        assert r.status_code == 200
        assert SECRET_DIR not in r.text
        assert r.json()["adapters"][0]["path"] == "style.gguf"

    def test_lora_list_keeps_paths_for_the_owner(self, temp_config, adapter):
        r = _client(OWNER).get("/api/lora")
        assert r.json()["adapters"][0]["path"] == f"{SECRET_DIR}/lora/style.gguf"


# ---------------------------------------------------------------------------
# 4 + 5. host hardware, credential status, Hub calls on the owner's token
# ---------------------------------------------------------------------------


class TestHubCallsAndHostDetails:
    @pytest.fixture
    def hub(self, monkeypatch):
        """Count Hub queries instead of making them."""
        from hfl.api import routes_discover, routes_draft, routes_recommend
        from hfl.hub.recommend import Recommendation

        calls: list[str] = []

        def recommend(**_):
            calls.append("recommend")
            return [
                Recommendation(
                    repo_id="org/m",
                    family="qwen",
                    quantization="q4_k_m",
                    parameter_estimate_b=7.0,
                    likes=1,
                    downloads=1,
                    license="apache-2.0",
                    gated=False,
                    estimated_vram_gb=4.1,
                    score=0.9,
                    reasoning=["fits comfortably (4.1/57.3 GB)"],
                )
            ]

        monkeypatch.setattr(routes_recommend, "recommend_models", recommend)
        monkeypatch.setattr(
            routes_draft, "pick_draft_for", lambda *a, **k: calls.append("draft") or None
        )
        monkeypatch.setattr(routes_discover, "search_hub", lambda q: calls.append("discover") or [])
        return calls

    @pytest.mark.parametrize(
        "path",
        ["/api/discover?refresh=true", "/api/recommend", "/api/draft/recommend?model=org/m"],
    )
    def test_a_remote_peer_cannot_force_hub_calls(self, temp_config, hub, path):
        r = _client(REMOTE).get(path)
        assert r.status_code == 403
        assert hub == []

    def test_cached_discovery_stays_public(self, temp_config, hub):
        # The owner fills the cache; a remote peer reads it without a Hub call.
        assert _client(OWNER).get("/api/discover?q=x&refresh=true").status_code == 200
        assert hub == ["discover"]
        r = _client(REMOTE).get("/api/discover?q=x")
        assert r.status_code == 200 and r.json()["cached"] is True
        assert hub == ["discover"]

    def test_the_owner_still_gets_the_host_profile(self, temp_config, hub):
        body = _client(OWNER).get("/api/recommend").json()
        assert body["hardware_profile"]["system_ram_gb"] is not None
        assert body["recommendations"][0]["reasoning"] == ["fits comfortably (4.1/57.3 GB)"]

    def test_remote_admin_does_not_see_host_hardware(self, temp_config, hub, monkeypatch):
        import hfl.config

        monkeypatch.setattr(hfl.config.config, "allow_remote_pull", True)
        r = _client(REMOTE).get("/api/recommend")
        assert r.status_code == 200
        body = r.json()
        assert body["hardware_profile"] is None
        assert body["recommendations"][0]["reasoning"] == ["fits comfortably"]
        assert "57.3" not in r.text

    def test_compliance_hides_credential_status_from_a_remote_peer(self, temp_config):
        remote = _client(REMOTE).get("/api/compliance/dashboard").json()
        assert "has_hf_token" not in remote and "gated_without_token" not in remote
        assert "total_models" in remote
        owner = _client(OWNER).get("/api/compliance/dashboard").json()
        assert "has_hf_token" in owner and "gated_without_token" in owner


# ---------------------------------------------------------------------------
# 6 + 7. image / speech: no paths for users, dispatcher, size cap
# ---------------------------------------------------------------------------


class _FakeDiffusers:
    calls: list[str] = []

    def load(self, model, *, local_files_only=False, **_):
        _FakeDiffusers.calls.append(model)

    def generate(self, prompt, **kw):
        return ImageResult(
            image_png_base64=base64.b64encode(b"PNG").decode(),
            width=kw["width"],
            height=kw["height"],
            seed=1,
            duration_s=0.1,
        )

    def unload(self):
        pass


class _FakeWhisper:
    calls: list[str] = []

    def load(self, model, *, local_files_only=False, **_):
        _FakeWhisper.calls.append(model)

    def transcribe(self, audio, language=None, include_segments=False):
        return WhisperResult(text="hi", language="en", duration_s=1.0, segments=None)

    def unload(self):
        pass


AUDIO = {"file": ("a.wav", b"RIFF....WAVE", "audio/wav")}


class TestMediaRoutes:
    @pytest.fixture(autouse=True)
    def fakes(self, monkeypatch, temp_config):
        from hfl.api import routes_images, routes_transcribe

        _FakeDiffusers.calls, _FakeWhisper.calls = [], []
        monkeypatch.setattr(routes_images, "is_available", lambda: True)
        monkeypatch.setattr(routes_images, "DiffusersEngine", _FakeDiffusers)
        monkeypatch.setattr(routes_transcribe, "is_available", lambda: True)
        monkeypatch.setattr(routes_transcribe, "WhisperEngine", _FakeWhisper)

    @pytest.mark.parametrize(
        "model", ["/etc", "~/.ssh/known_hosts", "../x", "./models/sd", "C:\\models\\sd", "a/../b"]
    )
    def test_a_remote_peer_cannot_name_a_path(self, model):
        remote = _client(REMOTE)
        r = remote.post("/api/images/generate", json={"model": model, "prompt": "x"})
        assert r.status_code == 400, r.text
        r = remote.post("/api/transcribe", files=AUDIO, data={"model": model})
        assert r.status_code == 400, r.text
        assert _FakeDiffusers.calls == [] and _FakeWhisper.calls == []

    def test_existence_is_not_revealed_for_relative_names(self, tmp_path, monkeypatch):
        # A repo-shaped name that exists relative to the server's cwd but
        # is not in the Hub cache: same 404 as one that does not exist.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "org" / "model").mkdir(parents=True)
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-cache"))
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        from hfl.hub.local_cache import hub_model_available_locally

        assert hub_model_available_locally("org/model") is False
        remote = _client(REMOTE)
        found = remote.post("/api/images/generate", json={"model": "org/model", "prompt": "x"})
        missing = remote.post("/api/images/generate", json={"model": "org/nope", "prompt": "x"})
        assert found.status_code == missing.status_code == 404

    def test_the_owner_may_still_use_a_path(self, tmp_path):
        owner = _client(OWNER)
        r = owner.post("/api/images/generate", json={"model": str(tmp_path), "prompt": "x"})
        assert r.status_code == 200
        r = owner.post("/api/transcribe", files=AUDIO, data={"model": str(tmp_path)})
        assert r.status_code == 200
        assert _FakeDiffusers.calls == [str(tmp_path)] and _FakeWhisper.calls == [str(tmp_path)]

    @pytest.mark.parametrize("size", ["99999x99999", "4097x512", "512x4097"])
    def test_image_size_is_capped(self, size):
        owner = _client(OWNER)
        r = owner.post(
            "/api/images/generate", json={"model": "org/sd", "prompt": "x", "size": size}
        )
        assert r.status_code == 422
        r = owner.post(
            "/v1/images/generations", json={"model": "org/sd", "prompt": "x", "size": size}
        )
        assert r.status_code == 400  # /v1 renders validation errors OpenAI's way
        assert _FakeDiffusers.calls == []

    def test_the_largest_allowed_size_passes(self):
        r = _client(OWNER).post(
            "/api/images/generate", json={"model": "org/sd", "prompt": "x", "size": "4096x4096"}
        )
        assert r.status_code == 200

    def test_media_goes_through_the_dispatcher(self):
        owner = _client(OWNER)
        before = _accepted()
        assert (
            owner.post("/api/images/generate", json={"model": "org/sd", "prompt": "x"}).status_code
            == 200
        )
        assert (
            owner.post(
                "/v1/images/generations", json={"model": "org/sd", "prompt": "x"}
            ).status_code
            == 200
        )
        assert owner.post("/api/transcribe", files=AUDIO).status_code == 200
        assert owner.post("/v1/audio/transcriptions", files=AUDIO).status_code == 200
        assert _accepted() == before + 4


class TestAFullQueueIsA429:
    """The new dispatcher paths answer a full queue like /api/chat does."""

    @pytest.fixture
    def full(self, monkeypatch):
        from hfl.api import helpers, routes_benchmark
        from hfl.engine.dispatcher import QueueFullError

        async def rejected(*_a, **_k):
            raise QueueFullError(depth=8, max_queued=8, retry_after=3)

        monkeypatch.setattr(helpers, "run_dispatched", rejected)
        monkeypatch.setattr(routes_benchmark, "run_dispatched", rejected)

    def test_media(self, temp_config, monkeypatch, full):
        from hfl.api import routes_images, routes_transcribe

        monkeypatch.setattr(routes_images, "is_available", lambda: True)
        monkeypatch.setattr(routes_transcribe, "is_available", lambda: True)
        owner = _client(OWNER)
        for r in (
            owner.post("/api/images/generate", json={"model": "org/sd", "prompt": "x"}),
            owner.post("/v1/images/generations", json={"model": "org/sd", "prompt": "x"}),
            owner.post("/api/transcribe", files=AUDIO),
            owner.post("/v1/audio/transcriptions", files=AUDIO),
        ):
            assert r.status_code == 429, r.text
            assert r.headers.get("retry-after") == "3"

    def test_diagnostics(self, engine, full):
        owner = _client(OWNER)
        assert owner.post("/api/verify/qwen-coder-7b").status_code == 429
        r = owner.post(
            "/api/benchmark/qwen-coder-7b",
            json={"prompt_lengths": [16], "runs_per_length": 1, "stream": False},
        )
        assert r.status_code == 429
