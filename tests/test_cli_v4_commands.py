# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Smoke + behaviour tests for the V4 CLI commands.

The Typer commands are thin wrappers over already-tested route /
engine code; these tests pin the wiring (registration, --help text,
exit codes for invalid input) without re-validating the underlying
behaviour.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from hfl.cli.main import app


@pytest.fixture
def runner():
    return CliRunner()


class TestCommandsRegistered:
    """Every V4 command must show up in ``hfl --help``."""

    def test_all_v4_commands_listed(self, runner):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in (
            "discover",
            "recommend",
            "lora",
            "pull-smart",
            "verify",
            "bench",
            "snapshot",
            "compliance-dashboard",
            "draft-recommend",
        ):
            assert cmd in result.stdout, f"{cmd} missing from main help"


class TestPullSmartCli:
    def test_planning_failure_exits_nonzero(self, runner, monkeypatch):
        from hfl.hub import smart_pull as src

        monkeypatch.setattr(
            src,
            "build_smart_plan",
            lambda *a, **kw: (_ for _ in ()).throw(ValueError("nothing fits")),
        )
        result = runner.invoke(app, ["pull-smart", "meta-llama/Llama-3.1-8B-Instruct"])
        assert result.exit_code == 1


class TestVerifyCli:
    def test_unknown_model_exits_nonzero(self, runner, monkeypatch, temp_config):
        result = runner.invoke(app, ["verify", "does-not-exist"])
        assert result.exit_code != 0


class TestBenchCli:
    def test_bad_lengths_exits_nonzero(self, runner):
        result = runner.invoke(
            app,
            ["bench", "any-model", "--lengths", "abc,xyz"],
        )
        assert result.exit_code == 1
        assert "Invalid --lengths" in result.stdout


class TestSnapshotCli:
    def test_unknown_action_exits_nonzero(self, runner):
        result = runner.invoke(app, ["snapshot", "explode"])
        assert result.exit_code == 1
        assert "save, load, list, delete" in result.stdout

    def test_save_without_name_exits_nonzero(self, runner):
        result = runner.invoke(app, ["snapshot", "save", "qwen-7b"])
        assert result.exit_code == 1

    def test_save_without_model_exits_nonzero(self, runner):
        result = runner.invoke(app, ["snapshot", "save", "", "--name", "warm-1"])
        assert result.exit_code == 1

    def test_list_empty_dashboard(self, runner, temp_config):
        # An empty snapshots directory: list should print "No snapshots
        # saved." and exit 0.
        result = runner.invoke(app, ["snapshot", "list"])
        assert result.exit_code == 0
        assert "No snapshots" in result.stdout

    def test_delete_missing_exits_nonzero(self, runner, temp_config):
        result = runner.invoke(app, ["snapshot", "delete", "--name", "nope"])
        assert result.exit_code == 1


class TestComplianceDashboardCli:
    def test_empty_registry_renders_dashboard(self, runner, temp_config):
        result = runner.invoke(app, ["compliance-dashboard"])
        assert result.exit_code == 0
        # Header + the two "By ..." sections always render.
        assert "Compliance dashboard" in result.stdout
        assert "total=0" in result.stdout


class TestDraftRecommendCli:
    def test_recommend_for_known_family_exits_0(self, runner, monkeypatch):
        from hfl.hub.draft_picker import DraftPick

        def _fake_pick(*args, **kwargs):
            return DraftPick(
                repo_id="meta-llama/Llama-3.2-1B-Instruct",
                family="llama",
                parameter_estimate_b=1.0,
                quantization=None,
                rationale="canonical fallback",
            )

        # The command imports inside the function — monkeypatch the
        # source module so the late import picks the fake.
        monkeypatch.setattr("hfl.hub.draft_picker.pick_draft_for", _fake_pick)

        result = runner.invoke(app, ["draft-recommend", "meta-llama/Llama-3.1-70B-Instruct"])
        assert result.exit_code == 0
        assert "Llama-3.2-1B-Instruct" in result.stdout

    def test_recommend_returns_nonzero_when_picker_returns_none(self, runner, monkeypatch):
        monkeypatch.setattr("hfl.hub.draft_picker.pick_draft_for", lambda *a, **kw: None)
        result = runner.invoke(app, ["draft-recommend", "anthropic/notamodel"])
        assert result.exit_code == 1
