# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the Modelfile ``REQUIRES`` check (Phase 7 P3-3)."""

from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import reset_state
from hfl.converter.requires_check import (
    InvalidRequiresError,
    RequiresNotSatisfiedError,
    check_requires,
    parse_requires,
)
from hfl.models.manifest import ModelManifest
from hfl.models.registry import get_registry, reset_registry

# ----------------------------------------------------------------------
# Unit tests for the parser / checker
# ----------------------------------------------------------------------


class TestParseRequires:
    def test_bare_version_upgrades_to_gte(self):
        spec = parse_requires("0.6.0")
        # ">=0.6.0" is what we synthesise.
        assert "0.6.1" in spec
        assert "0.5.9" not in spec

    def test_full_specifier(self):
        spec = parse_requires(">=0.6.0,<1.0")
        assert "0.9.9" in spec
        assert "1.0" not in spec

    def test_rejects_empty(self):
        with pytest.raises(InvalidRequiresError):
            parse_requires("")

    def test_rejects_malformed(self):
        with pytest.raises(InvalidRequiresError):
            parse_requires("lol")


class TestCheckRequires:
    def test_satisfied_is_silent(self):
        # Current version is guaranteed >= 0.1.
        check_requires(">=0.1.0")  # no raise

    def test_unsatisfied_raises(self):
        with pytest.raises(RequiresNotSatisfiedError):
            check_requires(">=99.0.0")

    def test_exception_carries_spec_and_current(self):
        with pytest.raises(RequiresNotSatisfiedError) as exc_info:
            check_requires(">=99.0.0")
        assert exc_info.value.spec == ">=99.0.0"
        # current comes from hfl.__version__; any non-empty string.
        assert exc_info.value.current

    def test_explicit_current_parameter(self):
        # With an explicit ``current``, the check becomes deterministic.
        check_requires(">=0.6.0", current="0.6.0")
        with pytest.raises(RequiresNotSatisfiedError):
            check_requires(">=0.7.0", current="0.6.0")


# ----------------------------------------------------------------------
# Route integration
# ----------------------------------------------------------------------


@pytest.fixture
def client(temp_config):
    reset_state()
    reset_registry()
    yield TestClient(app)
    reset_state()


def _parent(temp_config, name="llama3.3"):
    gguf = temp_config.home_dir / "models" / f"{name}.gguf"
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(b"fake gguf")
    m = ModelManifest(
        name=name,
        repo_id=f"org/{name}",
        local_path=str(gguf),
        format="gguf",
        size_bytes=gguf.stat().st_size,
        file_hash=hashlib.sha256(gguf.read_bytes()).hexdigest(),
    )
    get_registry().add(m)
    return m


class TestRouteGating:
    def test_satisfied_spec_allows_create(self, client, temp_config):
        _parent(temp_config)
        body = "FROM llama3.3\nREQUIRES >=0.1.0\n"
        resp = client.post(
            "/api/create",
            json={"model": "ok", "modelfile": body, "stream": False},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_unsatisfied_spec_rejects_with_400(self, client, temp_config):
        _parent(temp_config)
        body = "FROM llama3.3\nREQUIRES >=99.0.0\n"
        resp = client.post(
            "/api/create",
            json={"model": "nope", "modelfile": body, "stream": False},
        )
        assert resp.status_code == 400
        error = resp.json()["error"]
        assert "requires" in error.lower() or ">=99.0.0" in error

    def test_streaming_error_event_on_mismatch(self, client, temp_config):
        _parent(temp_config)
        body = "FROM llama3.3\nREQUIRES >=99.0.0\n"
        resp = client.post(
            "/api/create",
            json={"model": "nope", "modelfile": body, "stream": True},
        )
        events = [json.loads(line) for line in resp.text.strip().split("\n") if line.strip()]
        assert any("error" in e for e in events)
        err = [e for e in events if "error" in e][0]
        assert ">=99.0.0" in err["error"]

    def test_manifest_not_written_when_requires_fails(self, client, temp_config):
        _parent(temp_config)
        body = "FROM llama3.3\nREQUIRES >=99.0.0\n"
        client.post(
            "/api/create",
            json={"model": "should-not-exist", "modelfile": body, "stream": False},
        )
        assert get_registry().get("should-not-exist") is None
