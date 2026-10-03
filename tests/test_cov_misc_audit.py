# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The audit log: configuring it again keeps its sink, and a failure to
write never reaches the privileged operation being audited."""

from __future__ import annotations

import json
import logging

import pytest

from hfl.observability import audit


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv("HFL_AUDIT_LOG_PATH", raising=False)
    audit.reset_audit_log()
    yield
    audit.reset_audit_log()


def test_configuring_again_without_a_path_keeps_the_sink(tmp_path, monkeypatch) -> None:
    first = tmp_path / "audit.jsonl"
    audit.configure_audit_log(first)
    # Even with the env var now pointing elsewhere, an installed sink stays.
    monkeypatch.setenv("HFL_AUDIT_LOG_PATH", str(tmp_path / "other.jsonl"))
    audit.configure_audit_log()
    audit.audit_event("model.delete", resource="m")
    assert json.loads(first.read_text())["resource"] == "m"
    assert not (tmp_path / "other.jsonl").exists()


def test_a_failing_sink_is_logged_and_swallowed(tmp_path, monkeypatch, caplog) -> None:
    audit.configure_audit_log(tmp_path / "audit.jsonl")

    def broken(*a, **kw):
        raise OSError("disk full")

    assert audit._audit_logger is not None
    monkeypatch.setattr(audit._audit_logger, "info", broken)
    with caplog.at_level(logging.ERROR, logger="hfl.observability.audit"):
        audit.audit_event("model.pull", resource="m")  # must not raise
    assert "audit emission failed" in caplog.text


def test_metadata_is_copied_not_shared(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    audit.configure_audit_log(path)
    meta = {"parent": "a"}
    audit.audit_event("model.create", actor="api-key:1", resource="m", metadata=meta)
    meta["parent"] = "changed"
    row = json.loads(path.read_text())
    assert row["metadata"] == {"parent": "a"} and row["actor"] == "api-key:1"
    assert row["ts"].endswith("Z")
