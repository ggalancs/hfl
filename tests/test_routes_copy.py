# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the Ollama-compatible ``POST /api/copy`` endpoint and
the underlying ``ModelRegistry.copy`` method (Phase 5, P1-2).

The Ollama contract is specific about status codes — 200 OK on
success, 404 when the source is missing, 400 when the destination
is taken. Each case gets a dedicated test so client code that
branches on status stays stable.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.models.manifest import ModelManifest
from hfl.models.registry import ModelRegistry


@pytest.fixture
def client(temp_config):
    return TestClient(app)


@pytest.fixture
def two_models(temp_config, sample_manifest):
    """Register two models in the isolated temp-config registry.

    Depends on ``temp_config`` so the underlying registry file is
    fresh for every test — otherwise the real ``~/.hfl/models.json``
    leaks across the suite.
    """
    registry = ModelRegistry()
    registry.add(sample_manifest)  # The fixture's canonical manifest
    second = ModelManifest(
        name="other-model",
        repo_id="acme/other",
        local_path="/tmp/other.gguf",
        format="gguf",
    )
    registry.add(second)
    return registry, sample_manifest, second


# ----------------------------------------------------------------------
# Registry.copy unit tests
# ----------------------------------------------------------------------


class TestRegistryCopy:
    def test_copy_creates_second_entry(self, two_models):
        registry, src, _ = two_models
        assert registry.copy(src.name, "src-duplicate") is True

        duplicate = registry.get("src-duplicate")
        assert duplicate is not None
        assert duplicate.local_path == src.local_path
        # New entries drop the source's alias to avoid double-booking.
        assert duplicate.alias is None
        # Original still there
        assert registry.get(src.name) is not None

    def test_copy_fails_when_source_missing(self, two_models):
        registry, *_ = two_models
        assert registry.copy("ghost", "new") is False

    def test_copy_fails_when_destination_exists(self, two_models):
        registry, src, other = two_models
        assert registry.copy(src.name, other.name) is False

    def test_copy_rejects_invalid_destination_name(self, two_models):
        registry, src, _ = two_models
        from hfl.validators import ValidationError

        with pytest.raises(ValidationError):
            registry.copy(src.name, "../../etc/passwd")

    def test_copy_by_alias(self, two_models):
        registry, src, _ = two_models
        # Alias the source, then copy using the alias as source.
        registry.set_alias(src.name, "mymodel")
        assert registry.copy("mymodel", "via-alias") is True
        assert registry.get("via-alias") is not None


# ----------------------------------------------------------------------
# /api/copy route
# ----------------------------------------------------------------------


class TestRoutesCopy:
    def test_copy_returns_200_with_status(self, client, two_models):
        _, src, _ = two_models
        response = client.post("/api/copy", json={"source": src.name, "destination": "new-copy"})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "copied"
        assert body["source"] == src.name
        assert body["destination"] == "new-copy"

    def test_copy_source_not_found_is_404(self, client, two_models):
        response = client.post("/api/copy", json={"source": "phantom", "destination": "new-copy"})
        assert response.status_code == 404
        assert response.json().get("code") == "ModelNotFoundError"

    def test_copy_destination_taken_is_400(self, client, two_models):
        _, src, other = two_models
        response = client.post("/api/copy", json={"source": src.name, "destination": other.name})
        assert response.status_code == 400
        assert response.json().get("code") == "ModelAlreadyExistsError"

    def test_copy_malformed_destination_is_400(self, client, two_models):
        _, src, _ = two_models
        response = client.post(
            "/api/copy",
            json={"source": src.name, "destination": "../escape"},
        )
        assert response.status_code == 400

    def test_copy_empty_fields_rejected_at_422(self, client, two_models):
        # Pydantic enforces min_length=1 on both fields
        response = client.post("/api/copy", json={"source": "", "destination": "x"})
        assert response.status_code == 422

    def test_copy_preserves_registry_size_plus_one(self, client, two_models):
        """After /api/copy the registry has one more entry, and
        the new entry is reachable via ``registry.get``."""
        registry, src, _ = two_models
        response = client.post("/api/copy", json={"source": src.name, "destination": "new-copy"})
        assert response.status_code == 200
        # Use a fresh registry view to defeat any per-instance cache.
        fresh = ModelRegistry()
        assert fresh.get("new-copy") is not None
        assert fresh.get(src.name) is not None
