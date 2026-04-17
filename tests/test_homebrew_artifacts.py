# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Sanity tests for the Homebrew formula and its release workflow."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def _read(relpath: str) -> str:
    return (REPO / relpath).read_text(encoding="utf-8")


class TestFormula:
    def setup_method(self):
        self.text = _read("packaging/homebrew/hfl.rb")

    def test_class_name_matches_file(self):
        assert "class Hfl < Formula" in self.text

    def test_license_is_hrul(self):
        assert 'license "HRUL-1.0"' in self.text

    def test_virtualenv_install(self):
        assert "virtualenv_install_with_resources" in self.text

    def test_depends_on_python_3_12(self):
        assert 'depends_on "python@3.12"' in self.text

    def test_service_block_launches_hfl_serve(self):
        assert 'run [opt_bin/"hfl", "serve"' in self.text

    def test_has_smoke_test(self):
        assert "hfl version" in self.text
        assert "assert_match" in self.text


class TestWorkflow:
    def setup_method(self):
        self.cfg = yaml.safe_load(_read(".github/workflows/homebrew.yml"))

    def test_triggers_on_version_tags(self):
        on = self.cfg[True] if True in self.cfg else self.cfg["on"]
        assert on["push"]["tags"] == ["v*"]

    def test_waits_for_pypi_sdist(self):
        steps = self.cfg["jobs"]["update-tap"]["steps"]
        names = [s.get("name", "") for s in steps]
        assert any("Wait for PyPI" in n for n in names)

    def test_renders_formula(self):
        steps = self.cfg["jobs"]["update-tap"]["steps"]
        names = [s.get("name", "") for s in steps]
        assert any("Render formula" in n for n in names)

    def test_opens_tap_pr_conditionally(self):
        steps = self.cfg["jobs"]["update-tap"]["steps"]
        pr_step = next(s for s in steps if s.get("name", "").startswith("Open PR"))
        assert "HOMEBREW_TAP_TOKEN" in pr_step["if"]
