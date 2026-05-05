# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Integration tests for ``GET /api/compliance/dashboard`` (V4 F8)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import reset_state


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


def _add_manifest(*, name: str, license_id: str | None, gated: bool = False) -> None:
    from hfl.core.container import get_registry
    from hfl.models.manifest import ModelManifest

    manifest = ModelManifest(
        name=name,
        repo_id=f"org/{name}",
        local_path=f"/tmp/{name}",
        format="gguf",
        architecture="qwen",
        parameters="7B",
        license=license_id,
        gated=gated,
    )
    get_registry().add(manifest)


class TestComplianceDashboardShape:
    def test_empty_registry_returns_zeros(self, client):
        body = client.get("/api/compliance/dashboard").json()
        assert body["total_models"] == 0
        assert body["by_risk"] == {}
        assert body["by_license"] == {}
        assert body["gated_without_token"] == []
        assert body["missing_license"] == []
        assert body["eu_ai_act_warnings"] == []
        assert "has_hf_token" in body

    def test_envelope_keys(self, client):
        body = client.get("/api/compliance/dashboard").json()
        assert {
            "total_models",
            "by_risk",
            "by_license",
            "gated_without_token",
            "missing_license",
            "eu_ai_act_warnings",
            "has_hf_token",
        } <= body.keys()


class TestComplianceClassification:
    def test_permissive_license_is_categorised(self, client):
        _add_manifest(name="apache-model", license_id="apache-2.0")
        _add_manifest(name="mit-model", license_id="MIT")  # case-insensitive

        body = client.get("/api/compliance/dashboard").json()
        assert body["by_risk"]["permissive"] == 2

    def test_conditional_license_is_categorised(self, client):
        _add_manifest(name="llama-fork", license_id="llama3.1")

        body = client.get("/api/compliance/dashboard").json()
        assert body["by_risk"]["conditional"] == 1

    def test_unknown_license_is_categorised(self, client):
        _add_manifest(name="weird-fork", license_id="not-a-real-license")

        body = client.get("/api/compliance/dashboard").json()
        assert body["by_risk"]["unknown"] == 1

    def test_missing_license_appears_in_list(self, client):
        _add_manifest(name="undeclared", license_id=None)

        body = client.get("/api/compliance/dashboard").json()
        assert "undeclared" in body["missing_license"]
        assert body["by_risk"]["unknown"] == 1

    def test_by_license_counter(self, client):
        _add_manifest(name="m1", license_id="apache-2.0")
        _add_manifest(name="m2", license_id="apache-2.0")
        _add_manifest(name="m3", license_id="mit")

        body = client.get("/api/compliance/dashboard").json()
        assert body["by_license"]["apache-2.0"] == 2
        assert body["by_license"]["mit"] == 1


class TestGatedWithoutToken:
    def test_gated_model_without_token_is_flagged(self, client, monkeypatch):
        from hfl.config import config

        monkeypatch.setattr(config, "hf_token", None, raising=False)

        _add_manifest(name="meta-llama-3", license_id="llama3.1", gated=True)

        body = client.get("/api/compliance/dashboard").json()
        assert "meta-llama-3" in body["gated_without_token"]
        assert body["has_hf_token"] is False

    def test_gated_model_with_token_is_not_flagged(self, client, monkeypatch):
        from hfl.config import config

        monkeypatch.setattr(config, "hf_token", "hf_xxx", raising=False)

        _add_manifest(name="meta-llama-3", license_id="llama3.1", gated=True)

        body = client.get("/api/compliance/dashboard").json()
        assert body["gated_without_token"] == []
        assert body["has_hf_token"] is True


class TestEuAiActWarnings:
    def test_non_commercial_license_triggers_warning(self, client):
        _add_manifest(name="cc-bync", license_id="cc-by-nc-4.0")

        body = client.get("/api/compliance/dashboard").json()
        warnings = body["eu_ai_act_warnings"]
        assert len(warnings) == 1
        assert warnings[0]["model"] == "cc-bync"
        assert warnings[0]["license"] == "cc-by-nc-4.0"
        assert "EU AI Act" in warnings[0]["reason"]

    def test_permissive_license_does_not_trigger_warning(self, client):
        _add_manifest(name="apache-model", license_id="apache-2.0")

        body = client.get("/api/compliance/dashboard").json()
        assert body["eu_ai_act_warnings"] == []
