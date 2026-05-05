# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Integration tests for ``POST /api/pull/smart`` (V4 F2).

Pins the NDJSON event grammar and request/response shapes that the
CLI and external clients key on. The HF Hub and the underlying
``/api/pull`` machinery are mocked — the route's job is to plan,
emit ``planning`` / ``planned`` events and forward chunks; we verify
exactly that.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import reset_state


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


@pytest.fixture
def fake_smart_plan(monkeypatch):
    """Replace ``build_smart_plan`` with a predictable fake so the
    HF SDK is never reached during these tests."""
    from hfl.hub.smart_pull import SmartPullPlan

    plan = SmartPullPlan(
        target_repo_id="bartowski/Llama-3.1-8B-Instruct-GGUF",
        quantization="q5_k_m",
        estimated_vram_gb=7.8,
        reason="picked bartowski Q5_K_M (7.8 GB / 12.0 GB budget)",
        fallback_chain=["mlx-community/Llama-3.1-8B-Instruct-4bit: not on Hub"],
    )

    def _build(*args, **kwargs):
        return plan

    from hfl.api import routes_smart_pull as module

    monkeypatch.setattr(module, "build_smart_plan", _build, raising=False)

    # The function is imported lazily inside _stream_smart_pull; patch
    # the source module too so the late binding picks the fake.
    from hfl.hub import smart_pull as src

    monkeypatch.setattr(src, "build_smart_plan", _build)
    return plan


@pytest.fixture
def fake_pull_iter(monkeypatch):
    """Replace ``_iter_pull_events`` with a one-shot fake stream so we
    can assert the route forwards chunks verbatim."""

    async def _fake(model_name):
        yield json.dumps({"status": "downloading", "completed": 0}) + "\n"
        yield json.dumps({"status": "success"}) + "\n"

    from hfl.api import routes_pull as module

    monkeypatch.setattr(module, "_iter_pull_events", _fake, raising=False)
    return _fake


def _parse_ndjson(body: str) -> list[dict]:
    return [json.loads(line) for line in body.splitlines() if line.strip()]


# --- Happy path -------------------------------------------------------------


class TestSmartPullStreaming:
    def test_streaming_envelope_grammar(self, client, fake_smart_plan, fake_pull_iter):
        """Stream must emit: planning -> planned -> [pull events] in
        that order, with the right shape on each event."""
        response = client.post(
            "/api/pull/smart",
            json={"model": "meta-llama/Llama-3.1-8B-Instruct", "stream": True},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")

        events = _parse_ndjson(response.text)
        statuses = [e["status"] for e in events]
        # Required prefix: planning -> planned, then forwarded pull events.
        assert statuses[0] == "planning"
        assert statuses[1] == "planned"
        assert statuses[-1] == "success"

    def test_planned_event_carries_full_plan(self, client, fake_smart_plan, fake_pull_iter):
        events = _parse_ndjson(
            client.post(
                "/api/pull/smart",
                json={"model": "meta-llama/Llama-3.1-8B-Instruct"},
            ).text
        )
        planned = next(e for e in events if e["status"] == "planned")
        assert planned["target_repo_id"] == "bartowski/Llama-3.1-8B-Instruct-GGUF"
        assert planned["quantization"] == "q5_k_m"
        assert planned["estimated_vram_gb"] == 7.8
        assert "reason" in planned
        assert "fallback_chain" in planned

    def test_max_vram_gb_is_forwarded_to_planner(self, client, monkeypatch, fake_pull_iter):
        """When the request carries ``max_vram_gb`` the planner must
        receive it verbatim — operator-driven budget caps would
        silently no-op otherwise."""
        captured: dict = {}
        from hfl.hub.smart_pull import SmartPullPlan

        def _build(repo_id, *, profile=None, api=None, max_vram_gb=None):
            captured["max_vram_gb"] = max_vram_gb
            return SmartPullPlan(
                target_repo_id=repo_id,
                quantization="q4_k_m",
                estimated_vram_gb=5.0,
                reason="ok",
                fallback_chain=[],
            )

        from hfl.hub import smart_pull as src

        monkeypatch.setattr(src, "build_smart_plan", _build)

        client.post(
            "/api/pull/smart",
            json={"model": "x/y", "max_vram_gb": 6.0},
        )
        assert captured["max_vram_gb"] == 6.0


# --- Non-stream path --------------------------------------------------------


class TestSmartPullNonStream:
    def test_returns_last_event_as_json(self, client, fake_smart_plan, fake_pull_iter):
        response = client.post(
            "/api/pull/smart",
            json={"model": "meta-llama/Llama-3.1-8B-Instruct", "stream": False},
        )
        assert response.status_code == 200
        body = response.json()
        # The fake pull iterator emits 'success' as the last event.
        assert body["status"] == "success"

    def test_non_stream_planning_failure_returns_400(self, client, monkeypatch):
        """A planning failure (host too small, repo missing) on the
        non-stream path should map to HTTP 400 with the planner's
        message in the detail."""
        from hfl.hub import smart_pull as src

        def _boom(*args, **kwargs):
            raise ValueError("no variant of meta-llama/foo fits the 8.0 GB budget")

        monkeypatch.setattr(src, "build_smart_plan", _boom)

        response = client.post(
            "/api/pull/smart",
            json={"model": "meta-llama/foo", "stream": False},
        )
        assert response.status_code == 400
        assert "fits" in response.json()["detail"]


# --- Failure modes ----------------------------------------------------------


class TestSmartPullFailures:
    def test_streaming_planner_value_error_emits_failed(self, client, monkeypatch):
        from hfl.hub import smart_pull as src

        def _boom(*args, **kwargs):
            raise ValueError("no variant fits")

        monkeypatch.setattr(src, "build_smart_plan", _boom)

        response = client.post(
            "/api/pull/smart",
            json={"model": "meta-llama/foo", "stream": True},
        )
        assert response.status_code == 200  # NDJSON itself succeeds
        events = _parse_ndjson(response.text)
        assert events[-1]["status"] == "failed"
        assert "no variant fits" in events[-1]["error"]

    def test_streaming_planner_unexpected_exception_emits_failed(self, client, monkeypatch):
        from hfl.hub import smart_pull as src

        def _boom(*args, **kwargs):
            raise RuntimeError("hub network broken")

        monkeypatch.setattr(src, "build_smart_plan", _boom)

        events = _parse_ndjson(client.post("/api/pull/smart", json={"model": "x/y"}).text)
        assert events[-1]["status"] == "failed"
        # Generic failures are wrapped with a "planning failed" prefix
        # so the client can distinguish them from value errors.
        assert "planning failed" in events[-1]["error"]


# --- Validation -------------------------------------------------------------


class TestSmartPullValidation:
    def test_missing_model_field_is_422(self, client, fake_smart_plan):
        response = client.post("/api/pull/smart", json={})
        assert response.status_code in (400, 422)

    def test_negative_max_vram_gb_is_rejected(self, client, fake_smart_plan):
        response = client.post(
            "/api/pull/smart",
            json={"model": "x/y", "max_vram_gb": -1.0},
        )
        assert response.status_code in (400, 422)
