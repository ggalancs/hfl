# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the structured audit log (Phase 14 P2 — V2 row 27)."""

from __future__ import annotations

import json

import pytest

from hfl.observability import audit


@pytest.fixture(autouse=True)
def _fresh_audit():
    audit.reset_audit_log()
    yield
    audit.reset_audit_log()


def _lines(path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestNoOpWhenDisabled:
    def test_event_is_noop_when_path_unconfigured(self, monkeypatch):
        monkeypatch.delenv("HFL_AUDIT_LOG_PATH", raising=False)
        # Does not raise; writes nothing.
        audit.audit_event("model.create", resource="x")


class TestEmitsStructuredJSON:
    def test_basic_event(self, tmp_path, monkeypatch):
        log_path = tmp_path / "audit.log"
        monkeypatch.setenv("HFL_AUDIT_LOG_PATH", str(log_path))
        audit.audit_event(
            "model.create",
            actor="api-key:abcd1234",
            resource="coder",
            metadata={"parent": "llama3.3"},
        )
        entries = _lines(log_path)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["event"] == "model.create"
        assert entry["actor"] == "api-key:abcd1234"
        assert entry["resource"] == "coder"
        assert entry["metadata"] == {"parent": "llama3.3"}
        assert entry["outcome"] == "ok"
        assert entry["ts"].endswith("Z")

    def test_default_outcome_ok(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HFL_AUDIT_LOG_PATH", str(tmp_path / "a.log"))
        audit.audit_event("model.pull")
        entries = _lines(tmp_path / "a.log")
        assert entries[0]["outcome"] == "ok"

    def test_failure_outcome(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HFL_AUDIT_LOG_PATH", str(tmp_path / "a.log"))
        audit.audit_event("model.pull", outcome="error")
        entries = _lines(tmp_path / "a.log")
        assert entries[0]["outcome"] == "error"


class TestKnownEvents:
    def test_every_expected_event_is_registered(self):
        assert "model.create" in audit.AUDIT_EVENTS
        assert "blob.upload" in audit.AUDIT_EVENTS
        assert "api_key.mint" in audit.AUDIT_EVENTS

    def test_unknown_event_still_writes_but_warns(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("HFL_AUDIT_LOG_PATH", str(tmp_path / "a.log"))
        with caplog.at_level("WARNING"):
            audit.audit_event("not.registered", resource="x")
        entries = _lines(tmp_path / "a.log")
        assert len(entries) == 1
        assert any("not.registered" in r.message for r in caplog.records)


class TestRotation:
    def test_rotation_fires_when_max_bytes_reached(self, tmp_path):
        log_path = tmp_path / "audit.log"
        audit.configure_audit_log(
            path=str(log_path),
            max_bytes=200,
            backup_count=2,
        )
        # Each event is ~120 bytes, so 5 events → one rotation at least.
        for _ in range(5):
            audit.audit_event(
                "model.pull",
                resource="llama3.3",
                metadata={"source": "huggingface"},
            )
        # Primary log exists + at least one backup.
        backups = list(tmp_path.glob("audit.log.*"))
        assert len(backups) >= 1


class TestReconfigure:
    def test_reset_releases_handler(self, tmp_path):
        audit.configure_audit_log(path=str(tmp_path / "a.log"))
        audit.audit_event("model.pull")
        audit.reset_audit_log()
        # After reset, a second configure points to a new path cleanly.
        audit.configure_audit_log(path=str(tmp_path / "b.log"))
        audit.audit_event("model.delete")
        assert (tmp_path / "a.log").exists()
        assert (tmp_path / "b.log").exists()
