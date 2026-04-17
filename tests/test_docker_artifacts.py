# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Sanity tests for the Docker build artefacts.

We don't drive ``docker build`` from pytest (no daemon on CI's
test shard) — but the Dockerfile, .dockerignore, compose file and
the release workflow are worth parsing for regressions: typos in
image tags, missing healthchecks, unsigned pushes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent


def _read(p: str) -> str:
    return (REPO / p).read_text(encoding="utf-8")


class TestDockerfile:
    def setup_method(self):
        self.text = _read("Dockerfile")

    def test_uses_multistage_build(self):
        assert "AS builder" in self.text
        assert "AS runtime" in self.text

    def test_runs_as_non_root(self):
        assert "USER hfl" in self.text
        assert "useradd" in self.text

    def test_exposes_11434(self):
        assert "EXPOSE 11434" in self.text

    def test_has_healthcheck(self):
        assert "HEALTHCHECK" in self.text
        assert "/healthz" in self.text

    def test_entrypoint_is_tini(self):
        assert "ENTRYPOINT [\"/usr/bin/tini\"" in self.text

    def test_default_command_is_serve(self):
        assert "CMD [\"serve\"" in self.text

    def test_extras_are_parameterised(self):
        assert "ARG HFL_EXTRAS" in self.text
        assert 'pip install ".[${HFL_EXTRAS}]"' in self.text

    def test_pinned_python_version(self):
        assert "ARG PYTHON_VERSION=3.12" in self.text

    def test_hfl_home_is_var_lib(self):
        assert "HFL_HOME=/var/lib/hfl" in self.text


class TestDockerignore:
    def setup_method(self):
        self.lines = _read(".dockerignore").splitlines()

    def test_excludes_git(self):
        assert any(line.strip() == ".git" for line in self.lines)

    def test_excludes_models(self):
        assert any(line.strip() == "*.gguf" for line in self.lines)
        assert any(line.strip() == "*.safetensors" for line in self.lines)

    def test_excludes_venv(self):
        assert any(line.strip() == ".venv" for line in self.lines)
        assert any(line.strip() == ".venv-ci" for line in self.lines)

    def test_excludes_tests(self):
        assert any(line.strip() == "tests" for line in self.lines)


class TestDockerCompose:
    def setup_method(self):
        self.cfg = yaml.safe_load(_read("docker-compose.yml"))

    def test_single_service_named_hfl(self):
        assert list(self.cfg["services"].keys()) == ["hfl"]

    def test_port_mapping_11434(self):
        svc = self.cfg["services"]["hfl"]
        assert "${HFL_PORT:-11434}:11434" in svc["ports"]

    def test_persists_home_to_host(self):
        svc = self.cfg["services"]["hfl"]
        assert "./data/hfl:/var/lib/hfl" in svc["volumes"]

    def test_restart_policy(self):
        assert self.cfg["services"]["hfl"]["restart"] == "unless-stopped"

    def test_image_tag_parameterised(self):
        assert "${HFL_TAG:-latest}" in self.cfg["services"]["hfl"]["image"]


class TestDockerWorkflow:
    def setup_method(self):
        self.cfg = yaml.safe_load(_read(".github/workflows/docker.yml"))

    def test_triggers_on_version_tags_only(self):
        # Workflows default to running on every push unless we scope
        # to tags — this test locks that down so we don't accidentally
        # push on every main commit and burn GHCR quota.
        on_field = self.cfg[True] if True in self.cfg else self.cfg.get("on")
        push = on_field["push"]
        assert push["tags"] == ["v*"]
        assert "branches" not in push

    def test_builds_both_amd64_and_arm64(self):
        jobs = self.cfg["jobs"]["build-and-push"]
        steps = jobs["steps"]
        build_step = next(s for s in steps if s.get("id") == "build")
        assert "linux/amd64,linux/arm64" in build_step["with"]["platforms"]

    def test_has_cosign_sign_step(self):
        steps = self.cfg["jobs"]["build-and-push"]["steps"]
        names = {s.get("name") for s in steps}
        assert "Sign image (keyless)" in names

    def test_matrix_covers_llama_and_all_extras(self):
        matrix = self.cfg["jobs"]["build-and-push"]["strategy"]["matrix"]
        assert matrix["extras"] == ["llama", "all"]
