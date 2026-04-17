# SPDX-License-Identifier: HRUL-1.0
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

    def test_triggers_on_tag_only(self):
        on = self.cfg[True] if True in self.cfg else self.cfg["on"]
        assert on["push"]["tags"] == ["v*"]

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
