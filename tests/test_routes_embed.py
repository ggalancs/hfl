# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the embedding endpoints (P0-1).

Covers ``POST /api/embed``, ``POST /api/embeddings`` (legacy) and
``POST /v1/embeddings``. The real embedding backend is swapped out
for a deterministic fake so tests run without the ``[llama]`` or
``[transformers]`` extras installed.
"""

from __future__ import annotations

import base64
import struct
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state
from hfl.engine.embedding_engine import EmbeddingResult


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


@pytest.fixture
def fake_embed_engine(sample_manifest):
    """Register a fake embedding engine on state + registry.

    Bypasses the real model loader entirely so routes hit only the
    code we want to test.
    """
    from hfl.converter.formats import ModelType
    from hfl.models.registry import ModelRegistry

    sample_manifest.name = "nomic-embed-text"
    sample_manifest.model_type = "embedding"
    sample_manifest.architecture = "bert"
    sample_manifest.format = "gguf"
    ModelRegistry().add(sample_manifest)

    engine = MagicMock()
    engine.is_loaded = True

    def fake_embed(inputs, truncate=True, dimensions=None):
        vectors = [[float(i) + 0.1 * j for j in range(4)] for i, _ in enumerate(inputs)]
        if dimensions:
            vectors = [v[:dimensions] for v in vectors]
        return EmbeddingResult(
            embeddings=vectors,
            total_tokens=sum(len(s) for s in inputs),
            model=sample_manifest.name,
        )

    engine.embed = MagicMock(side_effect=fake_embed)

    with (
        patch("hfl.api.routes_embed.detect_model_type", return_value=ModelType.EMBEDDING)
        if False
        else patch(
            "hfl.converter.formats.detect_model_type",
            return_value=ModelType.EMBEDDING,
        ),
        patch(
            "hfl.api.routes_embed._select_embedding_backend",
            return_value=engine,
        ),
    ):
        # Warm the fake onto state so subsequent calls hit the fast path.
        state = get_state()
        state._embed_engine = engine
        state._embed_model_name = sample_manifest.name
        yield engine, sample_manifest


# --------------------------------------------------------------------
# /api/embed (Ollama preferred)
# --------------------------------------------------------------------


class TestOllamaEmbed:
    def test_single_string_input(self, client, fake_embed_engine):
        engine, manifest = fake_embed_engine
        response = client.post(
            "/api/embed",
            json={"model": manifest.name, "input": "hello world"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == manifest.name
        assert len(body["embeddings"]) == 1
        assert len(body["embeddings"][0]) == 4
        assert body["prompt_eval_count"] == len("hello world")
        assert body["total_duration"] > 0
        # load_duration is present even when fast-path hits (engine
        # already on state) — can be 0 but the field exists.
        assert "load_duration" in body

    def test_list_input_returns_vector_per_string(self, client, fake_embed_engine):
        _, manifest = fake_embed_engine
        response = client.post(
            "/api/embed",
            json={"model": manifest.name, "input": ["a", "bb", "ccc"]},
        )
        body = response.json()
        assert len(body["embeddings"]) == 3

    def test_dimensions_truncates(self, client, fake_embed_engine):
        _, manifest = fake_embed_engine
        response = client.post(
            "/api/embed",
            json={"model": manifest.name, "input": "x", "dimensions": 2},
        )
        body = response.json()
        assert len(body["embeddings"][0]) == 2

    def test_empty_list_rejected(self, client, fake_embed_engine):
        _, manifest = fake_embed_engine
        response = client.post(
            "/api/embed",
            json={"model": manifest.name, "input": []},
        )
        assert response.status_code == 422

    def test_oversized_batch_rejected(self, client, fake_embed_engine):
        _, manifest = fake_embed_engine
        response = client.post(
            "/api/embed",
            json={"model": manifest.name, "input": ["x"] * 2048},
        )
        assert response.status_code == 422

    def test_unknown_model_404(self, client):
        reset_state()
        response = client.post(
            "/api/embed",
            json={"model": "phantom/embed", "input": "hi"},
        )
        assert response.status_code == 404

    def test_keep_alive_honoured(self, client, fake_embed_engine):
        """keep_alive on /api/embed sets the deadline just like chat."""
        _, manifest = fake_embed_engine
        response = client.post(
            "/api/embed",
            json={"model": manifest.name, "input": "hi", "keep_alive": "10m"},
        )
        assert response.status_code == 200
        assert get_state().keep_alive_deadline_for(manifest.name) is not None


# --------------------------------------------------------------------
# /api/embeddings (legacy alias)
# --------------------------------------------------------------------


class TestOllamaEmbeddingsLegacy:
    def test_single_prompt_returns_single_vector(self, client, fake_embed_engine):
        _, manifest = fake_embed_engine
        response = client.post(
            "/api/embeddings",
            json={"model": manifest.name, "prompt": "hello"},
        )
        assert response.status_code == 200
        body = response.json()
        # Legacy envelope: single "embedding" field, not a list-of-lists.
        assert "embedding" in body
        assert isinstance(body["embedding"], list)
        assert len(body["embedding"]) == 4


# --------------------------------------------------------------------
# /v1/embeddings (OpenAI-compatible)
# --------------------------------------------------------------------


class TestOpenAIEmbeddings:
    def test_envelope_matches_openai(self, client, fake_embed_engine):
        _, manifest = fake_embed_engine
        response = client.post(
            "/v1/embeddings",
            json={"model": manifest.name, "input": "hello"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "list"
        assert body["model"] == manifest.name
        assert isinstance(body["data"], list)
        assert body["data"][0]["object"] == "embedding"
        assert body["data"][0]["index"] == 0
        assert "usage" in body
        assert "prompt_tokens" in body["usage"]

    def test_list_input_preserves_order(self, client, fake_embed_engine):
        _, manifest = fake_embed_engine
        response = client.post(
            "/v1/embeddings",
            json={"model": manifest.name, "input": ["first", "second"]},
        )
        body = response.json()
        indices = [e["index"] for e in body["data"]]
        assert indices == [0, 1]

    def test_base64_encoding(self, client, fake_embed_engine):
        """encoding_format=base64 must round-trip through
        little-endian floats so OpenAI SDK consumers can decode."""
        _, manifest = fake_embed_engine
        response = client.post(
            "/v1/embeddings",
            json={
                "model": manifest.name,
                "input": "x",
                "encoding_format": "base64",
            },
        )
        body = response.json()
        encoded = body["data"][0]["embedding"]
        assert isinstance(encoded, str)
        # Decode and verify it's a sequence of 4 little-endian floats
        raw = base64.b64decode(encoded)
        values = struct.unpack(f"<{len(raw) // 4}f", raw)
        assert len(values) == 4

    def test_unknown_encoding_format_rejected(self, client, fake_embed_engine):
        _, manifest = fake_embed_engine
        response = client.post(
            "/v1/embeddings",
            json={"model": manifest.name, "input": "x", "encoding_format": "int8"},
        )
        assert response.status_code == 422

    def test_token_list_input_decoded(self, client, fake_embed_engine):
        """``input=[1,2,3]`` is lossy but must not 500."""
        _, manifest = fake_embed_engine
        response = client.post(
            "/v1/embeddings",
            json={"model": manifest.name, "input": [1, 2, 3]},
        )
        assert response.status_code == 200
        body = response.json()
        # Token list collapses to a single "1 2 3" string → one vector.
        assert len(body["data"]) == 1

    def test_batch_of_token_lists(self, client, fake_embed_engine):
        _, manifest = fake_embed_engine
        response = client.post(
            "/v1/embeddings",
            json={"model": manifest.name, "input": [[1, 2], [3, 4]]},
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["data"]) == 2


# --------------------------------------------------------------------
# Model-type mismatch
# --------------------------------------------------------------------


class TestModelTypeMismatch:
    def test_llm_model_rejected_with_400(self, client, sample_manifest):
        """Asking an LLM to produce embeddings returns 400 with the
        ModelTypeMismatchError envelope, not a crashed 500."""
        from hfl.converter.formats import ModelType
        from hfl.models.registry import ModelRegistry

        reset_state()
        sample_manifest.name = "qwen-coder-7b"
        sample_manifest.format = "gguf"
        ModelRegistry().add(sample_manifest)

        with patch(
            "hfl.converter.formats.detect_model_type",
            return_value=ModelType.LLM,
        ):
            response = client.post(
                "/api/embed",
                json={"model": sample_manifest.name, "input": "hi"},
            )
        assert response.status_code == 400
        body = response.json()
        assert body.get("code") == "ModelTypeMismatchError"
