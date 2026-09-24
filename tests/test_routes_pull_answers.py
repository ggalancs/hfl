# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""/api/pull when the Hub answers "no": 404 or 403, not 500.

Found in the pre-release end-to-end run: pulling a repo that does not exist
answered 500 and logged a traceback at ERROR (already so in 0.20.0).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError


def _hub_error(cls, status):
    request = httpx.Request("GET", "https://huggingface.co/api/models/org/x")
    return cls(f"{status} from the Hub", response=httpx.Response(status, request=request))


@pytest.fixture
def owner(temp_config):
    from hfl.api.server import app

    return TestClient(app, client=("127.0.0.1", 50000))


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (
            lambda: _hub_error(RepositoryNotFoundError, 404),
            404,
            "not_found",
        ),
        (
            lambda: _hub_error(GatedRepoError, 403),
            403,
            "gated",
        ),
        (lambda: ValueError("Model not found: nothing-like-this"), 404, "not_found"),
    ],
)
def test_a_hub_no_is_the_callers_to_fix(owner, error, status, code, caplog):
    import logging

    with patch("hfl.hub.resolver.resolve", side_effect=error()):
        with caplog.at_level(logging.ERROR):
            response = owner.post("/api/pull", json={"model": "org/x", "stream": False})
    assert response.status_code == status
    assert response.json()["code"] == code
    assert "Traceback" not in caplog.text


def test_the_stream_carries_the_code(owner):
    with patch("hfl.hub.resolver.resolve", side_effect=_hub_error(RepositoryNotFoundError, 404)):
        response = owner.post("/api/pull", json={"model": "org/x", "stream": True})
    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert events[-1]["code"] == "not_found"


def test_gated_says_what_to_do(owner):
    with patch("hfl.hub.resolver.resolve", side_effect=_hub_error(GatedRepoError, 403)):
        body = owner.post("/api/pull", json={"model": "org/x", "stream": False}).json()
    assert "HF_TOKEN" in body["error"]


def test_anything_else_is_still_a_500_with_a_reference(owner):
    with patch("hfl.hub.resolver.resolve", side_effect=RuntimeError("/secret/path exploded")):
        response = owner.post("/api/pull", json={"model": "org/x", "stream": False})
    assert response.status_code == 500
    assert "/secret/path" not in response.text
