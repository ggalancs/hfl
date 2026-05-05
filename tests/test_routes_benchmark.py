# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Integration tests for ``POST /api/benchmark/{model}`` (V4 F3.2).

The benchmark engine itself is unit-tested in
``test_engine_benchmark.py``; these tests pin the HTTP wrapper:
NDJSON event grammar, request body defaults, stream/non-stream
paths, failure surfaces.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


@pytest.fixture
def llm_manifest():
    from hfl.models.manifest import ModelManifest

    return ModelManifest(
        name="qwen-coder-7b",
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        local_path="/tmp/qwen-coder-7b.gguf",
        format="gguf",
        architecture="qwen",
        parameters="7B",
    )


@pytest.fixture
def fast_engine(llm_manifest):
    """Mock a loaded engine whose generate calls return instantly so
    the test suite stays under a second per case."""
    state = get_state()
    engine = MagicMock(spec=["generate"])  # no streaming -> one path
    result = MagicMock()
    result.tokens_generated = 4
    result.text = "x" * 4
    engine.generate = MagicMock(return_value=result)
    state.engine = engine
    state.current_model = llm_manifest
    return engine


def _parse_ndjson(body: str) -> list[dict]:
    return [json.loads(line) for line in body.splitlines() if line.strip()]


class TestBenchmarkStreaming:
    def test_emits_full_event_grammar(self, client, llm_manifest, fast_engine):
        response = client.post(
            f"/api/benchmark/{llm_manifest.name}",
            json={"runs_per_length": 1, "prompt_lengths": [16], "max_tokens": 4},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")

        events = _parse_ndjson(response.text)
        statuses = [e["status"] for e in events]
        # Expected grammar: starting -> run(s) -> summary -> done.
        assert statuses[0] == "starting"
        assert "run" in statuses
        assert "summary" in statuses
        assert statuses[-1] == "done"

    def test_run_event_carries_metrics(self, client, llm_manifest, fast_engine):
        events = _parse_ndjson(
            client.post(
                f"/api/benchmark/{llm_manifest.name}",
                json={"runs_per_length": 1, "prompt_lengths": [16], "max_tokens": 4},
            ).text
        )
        run_events = [e for e in events if e["status"] == "run"]
        assert run_events
        for r in run_events:
            assert "ttft_ms" in r
            assert "total_ms" in r
            assert "tokens_per_second" in r
            assert "prompt_length" in r

    def test_summary_event_carries_p50_p95(self, client, llm_manifest, fast_engine):
        events = _parse_ndjson(
            client.post(
                f"/api/benchmark/{llm_manifest.name}",
                json={"runs_per_length": 3, "prompt_lengths": [16], "max_tokens": 4},
            ).text
        )
        summary = next(e for e in events if e["status"] == "summary")
        assert "ttft_p50_ms" in summary
        assert "ttft_p95_ms" in summary
        assert "tps_mean" in summary

    def test_default_body_uses_three_lengths(self, client, llm_manifest, fast_engine):
        """An empty body must use the documented defaults
        (16/256/2048, 3 runs, 64 max_tokens)."""
        response = client.post(f"/api/benchmark/{llm_manifest.name}", json={"max_tokens": 4})
        events = _parse_ndjson(response.text)
        starting = events[0]
        assert starting["prompt_lengths"] == [16, 256, 2048]
        assert starting["runs_per_length"] == 3


class TestBenchmarkNonStream:
    def test_returns_last_event_as_json(self, client, llm_manifest, fast_engine):
        response = client.post(
            f"/api/benchmark/{llm_manifest.name}",
            json={
                "runs_per_length": 1,
                "prompt_lengths": [16],
                "max_tokens": 4,
                "stream": False,
            },
        )
        assert response.status_code == 200
        body = response.json()
        # The last event of the stream is "done" — the route surfaces
        # it as the JSON body.
        assert body["status"] == "done"


class TestBenchmarkFailureSurfaces:
    def test_unknown_model_returns_400_or_404(self, client, monkeypatch):
        from hfl.api import routes_benchmark as module

        async def _missing(name):
            raise FileNotFoundError(f"model not found: {name}")

        monkeypatch.setattr(module, "load_llm", _missing)

        response = client.post(
            "/api/benchmark/does-not-exist",
            json={"runs_per_length": 1, "prompt_lengths": [16], "stream": False},
        )
        # FileNotFoundError surfaces in the failure event, which the
        # non-stream path reraises as 400.
        assert response.status_code in (400, 404)

    def test_engine_none_fails_gracefully(self, client, monkeypatch):
        from hfl.api import routes_benchmark as module

        async def _no_engine(name):
            return None, MagicMock(name=name)

        monkeypatch.setattr(module, "load_llm", _no_engine)

        response = client.post(
            "/api/benchmark/qwen",
            json={"runs_per_length": 1, "prompt_lengths": [16], "stream": True},
        )
        # Stream path returns 200 with a "failed" event as last line.
        events = _parse_ndjson(response.text)
        assert events[-1]["status"] == "failed"


class TestBenchmarkValidation:
    def test_runs_per_length_lower_bound(self, client, llm_manifest, fast_engine):
        response = client.post(
            f"/api/benchmark/{llm_manifest.name}",
            json={"runs_per_length": 0},
        )
        assert response.status_code in (400, 422)

    def test_max_tokens_upper_bound(self, client, llm_manifest, fast_engine):
        response = client.post(
            f"/api/benchmark/{llm_manifest.name}",
            json={"max_tokens": 999_999},
        )
        assert response.status_code in (400, 422)
