# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""RFC 8594 deprecation headers, on an endpoint that is actually deprecated.

`api/deprecation.py` implemented `add_deprecation_headers`,
`deprecated_endpoint` and `format_sunset_date`, and nothing was marked
deprecated, so the module had no subject. Meanwhile `/api/embeddings`
sits in the tree under a request model literally named
`OllamaEmbeddingsLegacyRequest`, superseded by `/api/embed` — deprecated
in every sense except the one a client can detect.

The choice that matters here is what is **not** sent. No `Sunset` date:
the endpoint has no removal date, and putting one in a header would be a
commitment nobody made, which a client would reasonably plan against.
`Deprecation: true` plus a `Link` to the successor says exactly what is
true — this is superseded, here is what replaces it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state


@pytest.fixture
def client(temp_config):
    from hfl.api.state import reset_state

    reset_state()
    yield TestClient(app)
    reset_state()


@pytest.fixture
def fake_embedder(monkeypatch):
    """Answer the embed call without loading a model."""
    from hfl.api import routes_embed

    engine = MagicMock()
    engine.embed = MagicMock(
        return_value=MagicMock(embeddings=[[0.1, 0.2]], total_tokens=2, model="m")
    )

    async def _load(model):
        return engine

    monkeypatch.setattr(routes_embed, "_load_embedding_model", _load)
    monkeypatch.setattr(routes_embed, "apply_keep_alive", lambda *a, **k: False)
    get_state()
    return engine


class TestTheLegacyEndpointAnnouncesItself:
    def test_it_sends_the_deprecation_header(self, client, fake_embedder):
        response = client.post("/api/embeddings", json={"model": "m", "prompt": "hi"})
        assert response.status_code == 200
        assert response.headers.get("Deprecation") == "true", (
            "a client calling the legacy endpoint gets no signal that it is "
            "superseded — which is how a deprecation goes unnoticed for years"
        )

    def test_it_points_at_the_successor(self, client, fake_embedder):
        response = client.post("/api/embeddings", json={"model": "m", "prompt": "hi"})
        link = response.headers.get("Link", "")
        assert "/api/embed" in link
        assert 'rel="successor-version"' in link, (
            "the Link header must say what kind of relation it is, or a client "
            "cannot act on it programmatically"
        )

    def test_it_still_works(self, client, fake_embedder):
        """Deprecated is not broken."""
        response = client.post("/api/embeddings", json={"model": "m", "prompt": "hi"})
        assert response.status_code == 200
        assert response.json()["embedding"] == [0.1, 0.2]


class TestNoPromiseIsMade:
    def test_no_sunset_date_is_sent(self, client, fake_embedder):
        """The header that would be a commitment.

        A Sunset date tells clients when the endpoint disappears. Nobody
        decided that, so sending one would invent a deadline the project
        has not agreed to and that users would plan migrations around.
        """
        response = client.post("/api/embeddings", json={"model": "m", "prompt": "hi"})
        assert "Sunset" not in response.headers, (
            "a removal date was announced that nobody committed to"
        )


class TestTheSupersedingEndpointIsUnaffected:
    def test_api_embed_carries_no_deprecation_headers(self, client, fake_embedder):
        response = client.post("/api/embed", json={"model": "m", "input": "hi"})
        assert response.status_code == 200
        assert "Deprecation" not in response.headers
        assert "Sunset" not in response.headers


class TestItIsDiscoverableBeforeYouCallIt:
    def test_openapi_marks_the_endpoint_deprecated(self, client):
        """Headers only reach someone who already called it.

        The OpenAPI flag is what shows a strikethrough in /docs and stops
        a new integration adopting the old endpoint in the first place.
        """
        schema = client.get("/openapi.json").json()
        legacy = schema["paths"]["/api/embeddings"]["post"]
        assert legacy.get("deprecated") is True
        assert schema["paths"]["/api/embed"]["post"].get("deprecated") is not True


class TestTheHelperItself:
    def test_a_sunset_date_is_formatted_as_an_http_date(self):
        """Unused here, but it is the module's contract and it is now
        reachable — so it gets held to RFC 7231 rather than left to rot."""
        from fastapi import Response

        from hfl.api.deprecation import add_deprecation_headers

        response = Response()
        add_deprecation_headers(response, sunset="2027-01-01T00:00:00")
        assert response.headers["Sunset"].endswith("GMT")
        assert "Jan 2027" in response.headers["Sunset"]

    def test_a_deprecation_date_becomes_a_unix_stamp(self):
        from fastapi import Response

        from hfl.api.deprecation import add_deprecation_headers

        response = Response()
        add_deprecation_headers(response, deprecated_at="2026-01-01T00:00:00")
        assert response.headers["Deprecation"].startswith("@")
