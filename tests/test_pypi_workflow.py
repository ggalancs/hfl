# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Sanity tests for the PyPI publish workflow."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def _load(relpath: str) -> dict:
    return yaml.safe_load((REPO / relpath).read_text(encoding="utf-8"))


class TestPyPIWorkflow:
    def setup_method(self):
        self.cfg = _load(".github/workflows/publish-pypi.yml")

    def test_is_manual_only(self):
        # Automatic CI/CD is disabled by owner policy: no auto-publish to PyPI on
        # tag push. The publish must be a deliberate manual workflow_dispatch
        # (the publish-pypi JOB still keeps its refs/tags/v guard so a manual run
        # only publishes when dispatched against a release tag).
        on = self.cfg[True] if True in self.cfg else self.cfg["on"]
        assert "workflow_dispatch" in on
        assert "push" not in on

    def test_has_oidc_id_token_permission(self):
        perms = self.cfg["permissions"]
        assert perms["id-token"] == "write"

    def test_build_job_builds_sdist_and_wheel(self):
        build = self.cfg["jobs"]["build"]
        run_steps = [step for step in build["steps"] if "run" in step]
        commands = " | ".join(step["run"] for step in run_steps)
        assert "python -m build" in commands
        assert "twine check" in commands

    def test_publish_pypi_is_tag_guarded(self):
        pub = self.cfg["jobs"]["publish-pypi"]
        # startsWith(github.ref, 'refs/tags/v') guard prevents main-branch pushes.
        assert "refs/tags/v" in pub["if"]

    def test_publish_pypi_uses_trusted_publisher_action(self):
        pub = self.cfg["jobs"]["publish-pypi"]
        names = [step.get("uses") for step in pub["steps"]]
        assert any(n and "pypa/gh-action-pypi-publish" in n for n in names)

    def test_testpypi_is_dispatch_only(self):
        # TestPyPI must not fire on tags — that would burn through
        # TestPyPI's version uniqueness quota.
        tpy = self.cfg["jobs"]["publish-testpypi"]
        assert "workflow_dispatch" in tpy["if"]

    def test_sigstore_attestation_attached(self):
        pub = self.cfg["jobs"]["publish-pypi"]
        names = [step.get("uses") for step in pub["steps"]]
        assert any(n and "sigstore" in n for n in names)
