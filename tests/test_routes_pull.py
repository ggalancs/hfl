# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the Ollama-compatible ``POST /api/pull`` endpoint.

Pins the NDJSON status stream shape — every status string
(``pulling manifest``, ``downloading``, ``verifying sha256 digest``,
``writing manifest``, ``success``, ``error``) and the event field
layout (``digest`` / ``total`` / ``completed``) is what Open WebUI
and ollama-python key off to render a progress bar.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app


@pytest.fixture
def client(temp_config):
    return TestClient(app)


@pytest.fixture
def mock_pull(tmp_path):
    """Patch the downloader + resolver into a predictable local file."""
    fake_file = tmp_path / "model.gguf"
    fake_file.write_bytes(b"\x00" * 1024)  # 1 KiB dummy content

    fake_resolved = MagicMock()
    fake_resolved.repo_id = "acme/test-model"
    fake_resolved.revision = "sha256:" + "a" * 64
    fake_resolved.format = "gguf"
    fake_resolved.filename = "model.gguf"

    with (
        patch("hfl.hub.resolver.resolve", return_value=fake_resolved) as m_resolve,
        patch("hfl.hub.downloader.pull_model", return_value=fake_file) as m_pull,
    ):
        yield m_resolve, m_pull, fake_file


def _parse_ndjson(body: bytes | str) -> list[dict]:
    """Split NDJSON body into a list of parsed events."""
    text = body if isinstance(body, str) else body.decode()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestPullStreaming:
    def test_happy_path_emits_ollama_sequence(self, client, mock_pull):
        """The canonical Ollama status sequence must be emitted in order."""
        response = client.post(
            "/api/pull",
            json={"model": "acme/test-model", "stream": True},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")

        events = _parse_ndjson(response.content)
        statuses = [e["status"] for e in events]

        # Minimum required sequence (in order). Additional
        # ``downloading`` heartbeats may appear in between.
        assert statuses[0] == "pulling manifest"
        assert "downloading" in statuses
        assert statuses[-3:] == [
            "verifying sha256 digest",
            "writing manifest",
            "success",
        ]

    def test_downloading_events_carry_digest_and_counters(self, client, mock_pull):
        response = client.post("/api/pull", json={"model": "acme/test-model", "stream": True})
        events = _parse_ndjson(response.content)
        downloads = [e for e in events if e["status"] == "downloading"]
        assert downloads, "Must emit at least one downloading event"

        for event in downloads:
            # Contract fields Open WebUI reads
            assert set(event.keys()) >= {"status", "digest", "total", "completed"}
            assert event["digest"].startswith("sha")
            assert isinstance(event["total"], int)
            assert isinstance(event["completed"], int)

        # Final "downloading" reports the actual on-disk bytes.
        final_dl = downloads[-1]
        assert final_dl["total"] == 1024  # 1 KiB dummy file
        assert final_dl["completed"] == final_dl["total"]

    def test_resolver_failure_emits_error_event(self, client, tmp_path):
        with patch(
            "hfl.hub.resolver.resolve",
            side_effect=ValueError("Model not found: ghost/phantom"),
        ):
            response = client.post("/api/pull", json={"model": "ghost/phantom", "stream": True})

        assert response.status_code == 200
        events = _parse_ndjson(response.content)
        statuses = [e["status"] for e in events]

        # Sequence starts with manifest, then jumps straight to error.
        assert statuses[0] == "pulling manifest"
        assert "error" in statuses
        error_event = [e for e in events if e["status"] == "error"][0]
        assert "ghost/phantom" in error_event["error"]

    def test_download_failure_emits_error_event(self, client, tmp_path):
        """If the download itself raises, the error line follows the
        opening downloading heartbeat."""
        fake_resolved = MagicMock()
        fake_resolved.repo_id = "acme/broken"
        fake_resolved.revision = None
        fake_resolved.format = "gguf"

        with (
            patch("hfl.hub.resolver.resolve", return_value=fake_resolved),
            patch(
                "hfl.hub.downloader.pull_model",
                side_effect=OSError("disk full"),
            ),
        ):
            response = client.post("/api/pull", json={"model": "acme/broken", "stream": True})

        events = _parse_ndjson(response.content)
        assert any(e["status"] == "error" for e in events)
        error = next(e for e in events if e["status"] == "error")
        assert "disk full" in error["error"]


class TestPullNonStreaming:
    def test_non_stream_returns_single_json_success(self, client, mock_pull):
        response = client.post("/api/pull", json={"model": "acme/test-model", "stream": False})
        assert response.status_code == 200
        # Response is pure JSON (not NDJSON).
        body = response.json()
        assert body["status"] == "success"
        # Duration is attached so scripts can log it.
        assert "_duration_seconds" in body

    def test_non_stream_returns_500_on_error(self, client):
        with patch("hfl.hub.resolver.resolve", side_effect=RuntimeError("boom")):
            response = client.post(
                "/api/pull",
                json={"model": "x/y", "stream": False},
            )
        assert response.status_code == 500
        body = response.json()
        assert body["status"] == "error"
        assert "boom" in body["error"]


class TestPullValidation:
    def test_empty_model_rejected(self, client):
        response = client.post("/api/pull", json={"model": "", "stream": False})
        assert response.status_code == 422

    def test_oversize_model_rejected(self, client):
        response = client.post("/api/pull", json={"model": "x" * 1024, "stream": False})
        assert response.status_code == 422

    def test_insecure_flag_accepted_but_ignored(self, client, mock_pull):
        """``insecure`` is part of Ollama's schema — we accept it for
        compatibility but it's a no-op (hf_hub downloads are always
        HTTPS)."""
        response = client.post(
            "/api/pull",
            json={"model": "acme/test-model", "insecure": True, "stream": False},
        )
        assert response.status_code == 200


class TestPullContractForOpenWebUI:
    """Black-box parity check: the event stream must be parseable by
    a naive NDJSON consumer that just iterates lines and reads the
    ``status`` field."""

    def test_every_line_is_parseable_json(self, client, mock_pull):
        response = client.post("/api/pull", json={"model": "acme/test-model", "stream": True})
        for line in response.content.decode().splitlines():
            if not line.strip():
                continue
            parsed = json.loads(line)
            assert "status" in parsed
