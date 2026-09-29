# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Sanity tests for Windows MSI + macOS DMG installer scaffolds."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


class TestWiXRecipe:
    def setup_method(self):
        self.text = _read("packaging/windows/hfl.wxs")

    def test_has_upgrade_code(self):
        assert "UpgradeCode=" in self.text

    def test_version_placeholder_present(self):
        assert "@HFL_VERSION@" in self.text

    def test_installs_per_machine(self):
        assert 'InstallScope="perMachine"' in self.text

    def test_installs_under_program_files_64(self):
        assert "ProgramFiles64Folder" in self.text

    def test_adds_install_dir_to_path(self):
        assert "PATH" in self.text
        assert 'Action="set"' in self.text


class TestWindowsWorkflow:
    def setup_method(self):
        self.cfg = yaml.safe_load(_read(".github/workflows/windows-msi.yml"))

    def test_is_manual_only(self):
        # Automatic CI/CD is disabled by owner policy: manual (workflow_dispatch)
        # only — no auto-build/publish on tag push.
        on = self.cfg[True] if True in self.cfg else self.cfg["on"]
        assert "workflow_dispatch" in on
        assert "push" not in on

    def test_signing_gated_on_secret(self):
        build = self.cfg["jobs"]["build-msi"]
        sign_step = next(s for s in build["steps"] if s.get("name") == "Sign MSI")
        assert "WINDOWS_CODE_SIGN_CERT" in sign_step["if"]

    def test_attaches_msi_to_release(self):
        steps = self.cfg["jobs"]["build-msi"]["steps"]
        names = [s.get("name", "") for s in steps]
        assert any("Attach to release" in n for n in names)


class TestMacOSWorkflow:
    def setup_method(self):
        self.cfg = yaml.safe_load(_read(".github/workflows/macos-dmg.yml"))

    def test_runs_on_apple_silicon_runner(self):
        job = self.cfg["jobs"]["build-dmg"]
        assert job["runs-on"] == "macos-14"

    def test_notarisation_gated_on_secret(self):
        job = self.cfg["jobs"]["build-dmg"]
        notary = next(s for s in job["steps"] if s.get("name") == "Notarise DMG")
        assert "MACOS_APPLE_ID" in notary["if"]

    def test_builds_dmg_with_create_dmg(self):
        job = self.cfg["jobs"]["build-dmg"]
        run_lines = " ".join(step.get("run", "") for step in job["steps"] if "run" in step)
        assert "create-dmg" in run_lines


class TestInstallersOnABranch:
    """A dispatch on a branch (the pre-release test on main) must end green:
    the DMG's "Attach to release" ran there and failed with "GitHub Releases
    requires a tag" (run 36491151024), while the MSI already skipped it."""

    WORKFLOWS = {".github/workflows/macos-dmg.yml": "build-dmg",
                 ".github/workflows/windows-msi.yml": "build-msi"}  # fmt: skip

    def _steps(self, rel: str) -> list[dict]:
        return yaml.safe_load(_read(rel))["jobs"][self.WORKFLOWS[rel]]["steps"]

    def test_the_release_is_attached_only_from_a_tag(self):
        for rel in self.WORKFLOWS:
            attach = [s for s in self._steps(rel) if "action-gh-release" in s.get("uses", "")]
            assert attach, rel
            for step in attach:
                assert "startsWith(github.ref, 'refs/tags/')" in step.get("if", ""), rel

    def test_the_build_is_kept_as_an_artifact_anyway(self):
        for rel in self.WORKFLOWS:
            assert any("upload-artifact" in s.get("uses", "") for s in self._steps(rel)), rel
