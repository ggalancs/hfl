# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""CLI edge security: tray exposure, container host networking, untrusted
manifest text in list/show, launch's API key off argv."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from hfl.cli import main as cli
from hfl.cli.main import app
from hfl.models.manifest import ModelManifest

runner = CliRunner()


@pytest.fixture(autouse=True)
def _unattended(monkeypatch):
    monkeypatch.delenv("HFL_API_KEY", raising=False)
    monkeypatch.delenv("HFL_ACCEPT_NETWORK_EXPOSURE", raising=False)
    monkeypatch.setattr(cli, "stdin_is_terminal", lambda: False)
    monkeypatch.setattr(cli, "_in_container", lambda: False)


# ----------------------------------------------------------------------
# Finding 4: serve --tray goes through the exposure check
# ----------------------------------------------------------------------


@pytest.mark.parametrize("how", ["flag", "env"])
def test_tray_does_not_bind_publicly_unattended_without_a_key(monkeypatch, how):
    trays: list = []
    monkeypatch.setattr(cli, "_run_tray", lambda *a, **k: trays.append(a))
    args = ["serve", "--tray"]
    if how == "flag":
        args += ["--host", "0.0.0.0"]
    else:  # HFL_HOST / OLLAMA_HOST resolve into config.host
        from hfl.config import config

        monkeypatch.setattr(config, "host", "0.0.0.0")
    result = runner.invoke(app, args)
    assert result.exit_code == 1, result.output
    assert trays == []
    assert "HFL_API_KEY" in result.output


def test_tray_still_starts_on_loopback_and_with_a_key(monkeypatch):
    trays: list = []
    monkeypatch.setattr(cli, "_run_tray", lambda *a, **k: trays.append(a))
    assert runner.invoke(app, ["serve", "--tray"]).exit_code == 0
    monkeypatch.setenv("HFL_API_KEY", "k")
    assert runner.invoke(app, ["serve", "--tray", "--host", "0.0.0.0"]).exit_code == 0
    assert len(trays) == 2


# ----------------------------------------------------------------------
# Finding 8: a container on the host's network is no isolated network
# ----------------------------------------------------------------------


def _netns(monkeypatch, ino: int | None, release: str = "7.0.12-linuxkit") -> str:
    real_stat = os.stat

    def _stat(path, *a, **k):
        if str(path) == "/proc/self/ns/net":
            if ino is None:
                raise PermissionError(path)
            return SimpleNamespace(st_ino=ino)
        return real_stat(path, *a, **k)

    with monkeypatch.context() as m:
        m.setattr(cli.sys, "platform", "linux")
        m.setattr(cli.os, "stat", _stat)
        m.setattr(cli.os, "uname", lambda: SimpleNamespace(release=release), raising=False)
        return cli._container_network()


def test_container_network_reads_the_namespace_inode(monkeypatch):
    # Measured with docker run on Linux 7.0: --network host vs the bridge.
    assert _netns(monkeypatch, 0xEFFFFFF9) == "host"
    assert _netns(monkeypatch, 4026532710) == "own"
    # Before 6.18 the host's inode was not fixed: no verdict either way.
    assert _netns(monkeypatch, 4026532710, release="6.8.0-1017-azure") == "unknown"
    assert _netns(monkeypatch, 0xEFFFFFF9, release="6.8.0") == "host"
    assert _netns(monkeypatch, None) == "unknown"


def _serve_in_container(monkeypatch, network: str):
    monkeypatch.setattr(cli, "_in_container", lambda: True)
    monkeypatch.setattr(cli, "_container_network", lambda: network)
    with patch("hfl.api.server.start_server") as start:
        result = runner.invoke(app, ["serve", "--host", "0.0.0.0"])
    return result, start


def test_host_networking_is_refused_like_any_unattended_public_bind(monkeypatch):
    result, start = _serve_in_container(monkeypatch, "host")
    assert result.exit_code == 1, result.output
    start.assert_not_called()
    assert "--network host" in result.output


def test_an_own_network_keeps_the_container_convenience(monkeypatch):
    result, start = _serve_in_container(monkeypatch, "own")
    assert result.exit_code == 0, result.output
    start.assert_called_once()
    assert "published" in result.output


def test_an_unverifiable_network_proceeds_but_says_so(monkeypatch):
    result, start = _serve_in_container(monkeypatch, "unknown")
    assert result.exit_code == 0, result.output
    start.assert_called_once()
    assert "Could not verify" in " ".join(result.output.split())


# ----------------------------------------------------------------------
# Finding 5: manifest text is not markup nor terminal control
# ----------------------------------------------------------------------

HOSTILE = "evil[/bold] \x1b]0;pwned\x07\x1b[31mred"


def _manifest() -> ModelManifest:
    return ModelManifest(
        name="m",
        repo_id="org/m",
        local_path="/nowhere/m.gguf",
        format="gguf",
        license=HOSTILE,
        license_name=HOSTILE,
        architecture=HOSTILE,
    )


def _registry(manifest):
    reg = MagicMock()
    reg.list_all.return_value = [manifest]
    reg.get.return_value = manifest
    return MagicMock(return_value=reg)


@pytest.mark.parametrize("args", [["list"], ["show", "m"], ["show", "m", "--license"]])
def test_hostile_manifest_text_is_printed_inert(args):
    with patch("hfl.models.registry.ModelRegistry", _registry(_manifest())):
        result = runner.invoke(app, args)
    assert result.exception is None or isinstance(result.exception, SystemExit), repr(
        result.exception
    )
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output and "\x07" not in result.output
    # Shown literally, not interpreted (the tag survives as text).
    assert "evil[/bold]" in result.output


# ----------------------------------------------------------------------
# Finding 7: launch passes the key through the environment
# ----------------------------------------------------------------------


def test_launch_starts_the_server_with_the_key_off_argv(monkeypatch, tmp_path):
    from hfl.cli.commands import launch as launcher

    seen: dict = {}

    def _popen(cmd, **kwargs):
        seen["cmd"], seen["env"] = cmd, kwargs["env"]
        return MagicMock(spec=subprocess.Popen)

    monkeypatch.setattr(launcher.subprocess, "Popen", _popen)
    launcher.start_server(11434, "s3cret-key", Path(tmp_path / "l.log"))
    assert "s3cret-key" not in " ".join(seen["cmd"])
    assert "--api-key" not in seen["cmd"]
    assert seen["env"]["HFL_API_KEY"] == "s3cret-key"


def test_launch_reads_hfl_api_key(monkeypatch):
    monkeypatch.setenv("HFL_API_KEY", "from-env")
    result = runner.invoke(app, ["launch", "claude", "-m", "m", "--print"])
    assert result.exit_code == 0, result.output
    assert "ANTHROPIC_AUTH_TOKEN=from-env" in result.output
