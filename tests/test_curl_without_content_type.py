# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``curl URL -d '{...}'`` works, as every example in Ollama's docs does.

Without ``-H 'Content-Type: application/json'`` curl labels the body a
form (``application/x-www-form-urlencoded``) and FastAPI refused it with
422. Ollama reads the body as JSON whatever the label says.

Only requests with no ``Origin`` header get that leniency. A browser
always sends ``Origin`` on a cross-origin POST, and requiring JSON is
what forces its CORS preflight: a web page posting a "simple" form to
localhost must keep being refused.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state

BODY = json.dumps({"model": "m", "stream": False}).encode()


@pytest.fixture
def client(temp_config, monkeypatch, sample_manifest):
    reset_state()
    state = get_state()
    state.engine = MagicMock(is_loaded=True)
    state.current_model = sample_manifest
    monkeypatch.setattr("hfl.api.routes_native._ensure_model_loaded", AsyncMock())
    yield TestClient(app)
    reset_state()


@pytest.mark.parametrize(
    "headers",
    [
        {"Content-Type": "application/x-www-form-urlencoded"},  # curl -d
        {"Content-Type": "text/plain"},
        {},
    ],
)
def test_a_json_body_is_read_as_json_without_the_label(client, headers):
    response = client.post("/api/generate", content=BODY, headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["done_reason"] == "load"


@pytest.mark.parametrize(
    "headers",
    [
        {"Content-Type": "application/x-www-form-urlencoded", "Origin": "https://evil.test"},
        {"Content-Type": "text/plain", "Origin": "null"},
    ],
)
def test_a_browser_form_post_is_still_refused(client, headers):
    response = client.post("/api/generate", content=BODY, headers=headers)
    assert response.status_code == 422


def _run(path: str, headers: list[tuple[bytes, bytes]], method: str = "POST") -> dict:
    from hfl.api.middleware import JSONBodyMiddleware

    seen: dict = {}

    async def inner(scope, receive, send):
        seen.update(scope)

    scope = {"type": "http", "method": method, "path": path, "headers": headers}
    asyncio.run(JSONBodyMiddleware(inner)(scope, AsyncMock(), AsyncMock()))
    return dict(seen["headers"])


def test_blob_uploads_keep_their_label():
    headers = _run("/api/blobs/sha256:abc", [(b"content-type", b"application/octet-stream")])
    assert headers[b"content-type"] == b"application/octet-stream"
    assert b"content-type" not in _run("/api/blobs/sha256:abc", [])


def test_multipart_and_get_are_left_alone():
    multipart = [(b"content-type", b"multipart/form-data; boundary=x")]
    assert _run("/v1/audio/transcriptions", multipart)[b"content-type"].startswith(b"multipart")
    assert b"content-type" not in _run("/api/tags", [], method="GET")
