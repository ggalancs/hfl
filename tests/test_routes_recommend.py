# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Integration tests for ``GET /api/recommend`` (V4 F1.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import reset_state


@dataclass
class _FakeCardData:
    license: str | None = None


@dataclass
class _FakeModelInfo:
    id: str
    likes: int = 0
    downloads: int = 0
    last_modified: datetime | None = None
    pipeline_tag: str | None = None
    library_name: str | None = None
    gated: bool = False
    tags: list[str] = field(default_factory=list)
    card_data: _FakeCardData | None = None


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


@pytest.fixture
def fake_hub(monkeypatch):
    """Inject a deterministic set of HF models — varied sizes and
    families so the recommender has something interesting to score."""
    import huggingface_hub

    api = MagicMock()
    api.list_models.return_value = iter(
        [
            _FakeModelInfo(
                id="meta-llama/Llama-3.1-8B-Instruct",
                likes=12000,
                downloads=8_000_000,
                last_modified=datetime.now(timezone.utc),
                pipeline_tag="text-generation",
                tags=["llama", "instruct"],
                card_data=_FakeCardData(license="llama3.1"),
            ),
            _FakeModelInfo(
                id="Qwen/Qwen2.5-7B-Instruct",
                likes=8000,
                downloads=4_000_000,
                tags=["qwen", "instruct"],
                card_data=_FakeCardData(license="apache-2.0"),
            ),
            _FakeModelInfo(
                id="meta-llama/Llama-3.1-70B-Instruct",
                likes=10000,
                downloads=2_000_000,
                tags=["llama", "instruct"],
            ),
            _FakeModelInfo(
                id="mlx-community/Llama-3.1-8B-Instruct-4bit",
                likes=400,
                downloads=80_000,
                tags=["mlx", "llama", "instruct"],
            ),
            _FakeModelInfo(
                id="Qwen/Qwen2.5-Coder-7B-Instruct",
                likes=3000,
                downloads=1_000_000,
                tags=["qwen", "code", "coder"],
            ),
        ]
    )
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    return api


@pytest.fixture
def cuda_24gb_profile(monkeypatch):
    """Pin the HW profile to a 24 GB CUDA host so recommendations
    are deterministic."""
    from hfl.api import routes_recommend as module
    from hfl.hub.hw_profile import HardwareProfile

    monkeypatch.setattr(
        module,
        "get_hw_profile",
        lambda: HardwareProfile(
            os="linux",
            arch="x86_64",
            system_ram_gb=64.0,
            gpu_kind="cuda",
            gpu_vram_gb=24.0,
            has_mlx=False,
            has_cuda=True,
            has_rocm=False,
        ),
    )


@pytest.fixture
def low_end_8gb_profile(monkeypatch):
    """Pin to a 16 GB total RAM, no-GPU host: the recommender should
    refuse 70B and prefer 7B/8B."""
    from hfl.api import routes_recommend as module
    from hfl.hub.hw_profile import HardwareProfile

    monkeypatch.setattr(
        module,
        "get_hw_profile",
        lambda: HardwareProfile(
            os="linux",
            arch="x86_64",
            system_ram_gb=16.0,
            gpu_kind="none",
            gpu_vram_gb=None,
            has_mlx=False,
            has_cuda=False,
            has_rocm=False,
        ),
    )


class TestRecommendShape:
    def test_returns_envelope_with_hardware_profile(self, client, fake_hub, cuda_24gb_profile):
        response = client.get("/api/recommend?task=chat&top_n=3")
        assert response.status_code == 200
        body = response.json()

        # Envelope keys.
        assert "hardware_profile" in body
        assert "recommendations" in body
        assert "task" in body
        assert body["task"] == "chat"
        assert body["total"] == len(body["recommendations"])

        # Hardware profile is the dataclass we patched.
        hp = body["hardware_profile"]
        assert hp["gpu_kind"] == "cuda"
        assert hp["gpu_vram_gb"] == 24.0

    def test_recommendation_carries_score_and_reasoning(self, client, fake_hub, cuda_24gb_profile):
        body = client.get("/api/recommend?task=chat&top_n=2").json()
        assert body["recommendations"]
        for rec in body["recommendations"]:
            assert "repo_id" in rec
            assert "score" in rec
            assert isinstance(rec["score"], (int, float))
            assert "reasoning" in rec
            assert isinstance(rec["reasoning"], list)
            assert "estimated_vram_gb" in rec

    def test_top_n_caps_results(self, client, fake_hub, cuda_24gb_profile):
        body = client.get("/api/recommend?top_n=2").json()
        assert len(body["recommendations"]) <= 2


class TestRecommendHardwareAware:
    def test_low_end_host_excludes_70b(self, client, fake_hub, low_end_8gb_profile):
        body = client.get("/api/recommend?top_n=10").json()
        ids = [r["repo_id"] for r in body["recommendations"]]
        assert all("70B" not in rid and "70b" not in rid for rid in ids)

    def test_cuda_host_can_include_70b(self, client, fake_hub, cuda_24gb_profile):
        # 70B Q4 ≈ 50 GB; even on 24 GB CUDA it overflows. Verify the
        # recommender excludes it on this host too.
        body = client.get("/api/recommend?top_n=10").json()
        ids = [r["repo_id"] for r in body["recommendations"]]
        assert all("70B" not in rid for rid in ids)


class TestRecommendValidation:
    def test_unknown_task_is_400(self, client, fake_hub, cuda_24gb_profile):
        response = client.get("/api/recommend?task=banana")
        assert response.status_code == 400

    def test_excessive_top_n_is_400(self, client, fake_hub, cuda_24gb_profile):
        response = client.get("/api/recommend?top_n=999")
        assert response.status_code in (400, 422)


class TestRecommendFailureSurfaces:
    def test_hub_failure_returns_503(self, client, monkeypatch, cuda_24gb_profile):
        from hfl.api import routes_recommend as module

        def _boom(**kwargs):
            raise RuntimeError("hub down")

        monkeypatch.setattr(module, "recommend_models", _boom)

        response = client.get("/api/recommend?task=chat")
        assert response.status_code == 503
