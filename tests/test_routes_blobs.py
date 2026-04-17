# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""HTTP contract tests for ``HEAD`` / ``POST /api/blobs/:digest``."""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import reset_state
from hfl.hub.blobs import blob_exists, blob_path


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestHeadBlob:
    def test_404_when_missing(self, client):
        digest = "a" * 64
        resp = client.head(f"/api/blobs/sha256:{digest}")
        assert resp.status_code == 404

    def test_200_when_present(self, client, temp_config):
        data = b"present blob"
        digest = _sha256(data)
        # Write directly to disk so we're testing only the HEAD route.
        blob_path(digest).write_bytes(data)
        resp = client.head(f"/api/blobs/sha256:{digest}")
        assert resp.status_code == 200

    def test_400_on_malformed_digest(self, client):
        resp = client.head("/api/blobs/sha256:bogus")
        assert resp.status_code == 400

    def test_accepts_bare_hex_without_prefix(self, client, temp_config):
        data = b"no prefix"
        digest = _sha256(data)
        blob_path(digest).write_bytes(data)
        resp = client.head(f"/api/blobs/{digest}")
        assert resp.status_code == 200


class TestPostBlob:
    def test_happy_path_returns_201(self, client, temp_config):
        data = b"abcdefghij" * 100
        digest = _sha256(data)
        resp = client.post(f"/api/blobs/sha256:{digest}", content=data)
        assert resp.status_code == 201
        assert resp.headers["X-Blob-Bytes"] == str(len(data))
        assert blob_exists(digest)
        assert blob_path(digest).read_bytes() == data

    def test_digest_mismatch_returns_400(self, client, temp_config):
        data = b"payload"
        wrong = "0" * 64
        resp = client.post(f"/api/blobs/sha256:{wrong}", content=data)
        assert resp.status_code == 400
        assert "mismatch" in resp.json()["detail"].lower()
        assert not blob_exists(wrong)

    def test_malformed_digest_returns_400(self, client, temp_config):
        resp = client.post("/api/blobs/sha256:not-hex", content=b"anything")
        assert resp.status_code == 400

    def test_empty_body_accepted_when_matches(self, client, temp_config):
        empty_digest = _sha256(b"")
        resp = client.post(f"/api/blobs/sha256:{empty_digest}", content=b"")
        assert resp.status_code == 201
        assert blob_path(empty_digest).read_bytes() == b""

    def test_large_body_bypasses_request_body_limit(self, client, temp_config):
        # Even when the global body-size cap is tiny, the /api/blobs
        # prefix is whitelisted in RequestBodyLimitMiddleware, so the
        # route still accepts the payload.
        from hfl.api.middleware import RequestBodyLimitMiddleware

        assert "/api/blobs/" in RequestBodyLimitMiddleware.EXCLUDED_PREFIXES
        data = b"x" * (256 * 1024)  # 256 KiB — well over typical text caps
        digest = _sha256(data)
        resp = client.post(f"/api/blobs/sha256:{digest}", content=data)
        assert resp.status_code == 201
