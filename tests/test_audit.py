# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for audit logging and compliance reporting."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hfl.security import AuditEvent, audit


# =============================================================================
# AuditEvent tests
# =============================================================================


class TestAuditEventCreation:
    """Tests for AuditEvent dataclass creation."""

    def test_audit_event_creation(self):
        """Test creating an AuditEvent with all fields."""
        event = AuditEvent(
            event_type="MODEL_ACCESS",
            action="chat_completion",
            timestamp="2026-03-07T12:00:00+00:00",
            client_ip="127.0.0.1",
            user_id="user-123",
            model="llama3-q4",
            details={"tokens": 512},
        )
        assert event.event_type == "MODEL_ACCESS"
        assert event.action == "chat_completion"
        assert event.timestamp == "2026-03-07T12:00:00+00:00"
        assert event.client_ip == "127.0.0.1"
        assert event.user_id == "user-123"
        assert event.model == "llama3-q4"
        assert event.details == {"tokens": 512}

    def test_audit_event_default_timestamp(self):
        """Test that AuditEvent generates a timestamp when not provided."""
        before = datetime.now(timezone.utc)
        event = AuditEvent(event_type="AUTH_FAILURE", action="login")
        after = datetime.now(timezone.utc)

        assert event.timestamp != ""
        # Parse and validate the generated timestamp
        parsed = datetime.fromisoformat(event.timestamp)
        assert before <= parsed <= after

    def test_audit_event_defaults(self):
        """Test AuditEvent default values for optional fields."""
        event = AuditEvent(event_type="MODEL_LOAD", action="load")
        assert event.client_ip == ""
        assert event.user_id is None
        assert event.model is None
        assert event.details == {}

    def test_audit_event_preserves_explicit_timestamp(self):
        """Test that an explicit timestamp is preserved."""
        ts = "2025-01-01T00:00:00+00:00"
        event = AuditEvent(event_type="TEST", action="test", timestamp=ts)
        assert event.timestamp == ts

    def test_audit_event_empty_details_not_shared(self):
        """Test that each event gets its own details dict."""
        event1 = AuditEvent(event_type="A", action="a")
        event2 = AuditEvent(event_type="B", action="b")
        event1.details["key"] = "value"
        assert "key" not in event2.details


# =============================================================================
# audit() function tests
# =============================================================================


class TestAuditFunction:
    """Tests for the audit() logging function."""

    def test_audit_function_logs(self, caplog):
        """Test that audit() logs the event at INFO level."""
        event = AuditEvent(
            event_type="MODEL_DELETE",
            action="rm",
            model="test-model",
            client_ip="10.0.0.1",
            user_id="admin",
            details={"reason": "cleanup"},
        )

        with caplog.at_level(logging.INFO, logger="hfl.audit"):
            audit(event)

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert "MODEL_DELETE" in record.message
        assert "rm" in record.message
        assert record.timestamp == event.timestamp
        assert record.event_type == "MODEL_DELETE"
        assert record.user_id == "admin"
        assert record.client_ip == "10.0.0.1"
        assert record.model == "test-model"
        assert record.action == "rm"
        assert record.details == {"reason": "cleanup"}

    def test_audit_function_logs_minimal_event(self, caplog):
        """Test audit() with minimal event (only required fields)."""
        event = AuditEvent(event_type="INFO", action="healthcheck")

        with caplog.at_level(logging.INFO, logger="hfl.audit"):
            audit(event)

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert "INFO" in record.message
        assert "healthcheck" in record.message


# =============================================================================
# Compliance Report tests
# =============================================================================


def _make_mock_manifest(
    name: str = "test-model",
    repo_id: str = "org/test-model",
    license: str = "apache-2.0",
    local_path: str = "/models/test",
    created_at: str = "2026-01-01T00:00:00",
    alias: str | None = None,
):
    """Create a mock ModelManifest for testing."""
    manifest = MagicMock()
    manifest.name = name
    manifest.repo_id = repo_id
    manifest.license = license
    manifest.local_path = local_path
    manifest.created_at = created_at
    manifest.alias = alias
    return manifest


class TestComplianceReport:
    """Tests for the compliance-report CLI command."""

    @patch("hfl.models.registry.get_registry")
    def test_compliance_report_json_output(self, mock_get_registry, tmp_path):
        """Test generating a JSON compliance report."""
        from typer.testing import CliRunner

        from hfl.cli.main import app

        # Setup mock registry
        mock_registry = MagicMock()
        mock_registry.list_all.return_value = [
            _make_mock_manifest(
                name="llama3-q4",
                repo_id="meta-llama/Llama-3",
                license="llama3",
                local_path="/models/llama3",
                created_at="2026-02-01T10:00:00",
                alias="llama",
            ),
            _make_mock_manifest(
                name="mistral-7b",
                repo_id="mistralai/Mistral-7B",
                license="apache-2.0",
                local_path="/models/mistral",
                created_at="2026-01-15T08:30:00",
            ),
        ]
        mock_get_registry.return_value = mock_registry

        output_file = tmp_path / "report.json"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["compliance-report", "--output", str(output_file), "--format", "json"],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output_file.exists()

        report = json.loads(output_file.read_text())
        assert report["hfl_version"] == "0.1.0"
        assert report["total_models"] == 2
        assert len(report["models"]) == 2

        # Check first model
        m1 = report["models"][0]
        assert m1["name"] == "llama3-q4"
        assert m1["repo_id"] == "meta-llama/Llama-3"
        assert m1["license"] == "llama3"
        assert m1["alias"] == "llama"

        # Check second model (no alias)
        m2 = report["models"][1]
        assert m2["name"] == "mistral-7b"
        assert "alias" not in m2

    @patch("hfl.models.registry.get_registry")
    def test_compliance_report_markdown_output(self, mock_get_registry, tmp_path):
        """Test generating a Markdown compliance report."""
        from typer.testing import CliRunner

        from hfl.cli.main import app

        mock_registry = MagicMock()
        mock_registry.list_all.return_value = [
            _make_mock_manifest(
                name="phi-3",
                repo_id="microsoft/phi-3",
                license="mit",
                local_path="/models/phi3",
                created_at="2026-03-01T00:00:00",
            ),
        ]
        mock_get_registry.return_value = mock_registry

        output_file = tmp_path / "report.md"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["compliance-report", "--output", str(output_file), "--format", "markdown"],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output_file.exists()

        content = output_file.read_text()
        assert "# HFL Compliance Report" in content
        assert "Total models: 1" in content
        assert "## phi-3" in content
        assert "- Repository: microsoft/phi-3" in content
        assert "- License: mit" in content
        assert "- Path: /models/phi3" in content

    @patch("hfl.models.registry.get_registry")
    def test_compliance_report_empty_registry(self, mock_get_registry, tmp_path):
        """Test compliance report with no models."""
        from typer.testing import CliRunner

        from hfl.cli.main import app

        mock_registry = MagicMock()
        mock_registry.list_all.return_value = []
        mock_get_registry.return_value = mock_registry

        output_file = tmp_path / "empty_report.json"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["compliance-report", "--output", str(output_file)],
        )

        assert result.exit_code == 0
        report = json.loads(output_file.read_text())
        assert report["total_models"] == 0
        assert report["models"] == []
