# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl launch``: open Claude Code or Codex on a local model.

The settings were verified against the real tools before being written
down here: Claude Code 2.1.282 fixed a bug through /v1/messages and
Codex 0.156.1 through /v1/responses, both on a local Qwen3-Coder.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from hfl.cli.commands import launch as launcher

BASE = "http://127.0.0.1:11434"


class TestSettings:
    def test_claude(self):
        plan = launcher.build_launch(
            "claude", BASE, "qwen-coder", context=65536, extra=["-p", "hi"]
        )
        assert plan.argv == ["claude", "-p", "hi"]
        assert plan.env == {
            "ANTHROPIC_BASE_URL": BASE,
            "ANTHROPIC_AUTH_TOKEN": "hfl",
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_MODEL": "qwen-coder",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen-coder",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen-coder",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen-coder",
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "65536",
        }

    def test_claude_with_a_key_and_no_known_context(self):
        env = launcher.build_launch("claude", BASE, "m", api_key="s3cret").env
        assert env["ANTHROPIC_AUTH_TOKEN"] == "s3cret"
        assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in env

    def test_codex(self):
        plan = launcher.build_launch(
            "codex", BASE, "qwen-coder", context=32768, extra=["exec", "x"]
        )
        assert plan.argv == [
            "codex",
            "-c",
            "model_provider=hfl",
            "-c",
            f'model_providers.hfl={{name="HFL", base_url="{BASE}/v1", wire_api="responses"}}',
            "-c",
            "model_context_window=32768",
            "-m",
            "qwen-coder",
            "exec",
            "x",
        ]
        assert plan.env == {}

    def test_codex_with_a_key(self):
        plan = launcher.build_launch("codex", BASE, "m", api_key="s3cret")
        assert 'env_key="HFL_API_KEY"' in plan.argv[4]
        assert plan.env == {"HFL_API_KEY": "s3cret"}

    def test_an_unknown_tool(self):
        with pytest.raises(launcher.LaunchError):
            launcher.build_launch("vim", BASE, "m")

    def test_shell_lines_are_quoted(self):
        lines = launcher.shell_lines(launcher.Launch(argv=["claude"], env={"A": "x y", "B": ""}))
        assert lines == ["export A='x y'", "export B=''", "claude"]


@pytest.fixture
def world(monkeypatch):
    """Everything ``run`` touches outside itself, recorded."""
    calls: dict = {"started": 0, "stopped": 0, "ran": None}
    state = {"up": True, "tool_code": 0, "preload_error": None}

    monkeypatch.setattr(launcher.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(launcher, "server_up", lambda url: state["up"])

    def start(port, api_key, log_path):
        calls["started"] += 1
        return MagicMock(spec=subprocess.Popen)

    def stop(proc):
        calls["stopped"] += 1

    def preload(url, model, key):
        if state["preload_error"]:
            raise launcher.LaunchError(state["preload_error"])
        return 40960

    def run_tool(argv, env):
        calls["ran"] = (argv, env)
        if isinstance(state["tool_code"], BaseException):
            raise state["tool_code"]
        return MagicMock(returncode=state["tool_code"])

    monkeypatch.setattr(launcher, "start_server", start)
    monkeypatch.setattr(launcher, "stop_server", stop)
    monkeypatch.setattr(launcher, "wait_until_up", lambda url, proc: True)
    monkeypatch.setattr(launcher, "preload", preload)
    monkeypatch.setattr(launcher.subprocess, "run", run_tool)
    return calls, state


def _run(tool="claude", host="127.0.0.1"):
    return launcher.run(
        tool,
        "m",
        host=host,
        port=11434,
        api_key=None,
        extra=[],
        log_path=Path("/nowhere/launch.log"),
        say=lambda message: None,
    )


class TestRun:
    def test_a_running_server_is_reused(self, world):
        calls, state = world
        assert _run() == 0
        assert calls["started"] == 0 and calls["stopped"] == 0
        argv, env = calls["ran"]
        assert argv == ["claude"] and env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "40960"

    def test_a_server_it_starts_is_stopped_and_the_exit_code_passes_through(self, world):
        calls, state = world
        state["up"], state["tool_code"] = False, 3
        assert _run() == 3
        assert calls["started"] == 1 and calls["stopped"] == 1

    def test_the_server_is_stopped_even_when_the_tool_blows_up(self, world):
        calls, state = world
        state["up"], state["tool_code"] = False, KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            _run()
        assert calls["stopped"] == 1

    def test_a_model_that_cannot_load_stops_before_the_tool(self, world):
        calls, state = world
        state["up"], state["preload_error"] = False, "HTTP 507: Not enough memory"
        with pytest.raises(launcher.LaunchError, match="507"):
            _run()
        assert calls["ran"] is None and calls["stopped"] == 1

    def test_a_remote_host_is_never_started_here(self, world):
        calls, state = world
        state["up"] = False
        with pytest.raises(launcher.LaunchError):
            _run(host="10.0.0.5")
        assert calls["started"] == 0

    def test_a_missing_tool_fails_first(self, world, monkeypatch):
        calls, _ = world
        monkeypatch.setattr(launcher.shutil, "which", lambda tool: None)
        monkeypatch.setattr(launcher, "server_up", lambda url: pytest.fail("touched the server"))
        with pytest.raises(launcher.LaunchError, match="not installed"):
            _run()


class TestPreload:
    def test_it_returns_the_loaded_models_context(self, monkeypatch):
        import httpx

        monkeypatch.setattr(httpx, "post", lambda *a, **k: MagicMock(status_code=200))
        ps = {"models": [{"name": "m", "model": "m", "details": {"context_size": 131072}}]}
        monkeypatch.setattr(httpx, "get", lambda *a, **k: MagicMock(json=lambda: ps))
        assert launcher.preload(BASE, "m", None) == 131072

    def test_the_servers_words_on_failure(self, monkeypatch):
        import httpx

        body = {"error": "Not enough memory to load m on this server."}
        response = MagicMock(status_code=507, json=lambda: body)
        monkeypatch.setattr(httpx, "post", lambda *a, **k: response)
        with pytest.raises(launcher.LaunchError, match="Not enough memory"):
            launcher.preload(BASE, "m", None)


class TestCommand:
    @pytest.fixture
    def cli(self, temp_config):
        from hfl.cli.main import app

        return lambda *args: CliRunner().invoke(app, ["launch", *args])

    def test_print(self, cli):
        result = cli("claude", "-m", "qwen-coder", "--print")
        assert result.exit_code == 0
        assert "export ANTHROPIC_BASE_URL=http://127.0.0.1:11434" in result.stdout
        assert result.stdout.strip().endswith("claude")

    def test_a_model_is_required(self, cli):
        assert cli("claude").exit_code == 2

    def test_a_missing_tool_is_reported_before_any_download(self, cli, monkeypatch):
        import hfl.cli.main as main

        monkeypatch.setattr(launcher.shutil, "which", lambda tool: None)
        monkeypatch.setattr(
            main, "_local_or_pulled", lambda *a: pytest.fail("resolved the model first")
        )
        result = cli("codex", "-m", "org/model:Q4_K_M")
        assert result.exit_code == 1 and "not installed" in result.stdout

    def test_an_unknown_model(self, cli, monkeypatch):
        monkeypatch.setattr(launcher.shutil, "which", lambda tool: "/usr/bin/claude")
        result = cli("claude", "-m", "nope")
        assert result.exit_code == 1 and "not a local model" in result.stdout

    def test_extra_arguments_reach_the_tool(self, cli, monkeypatch, temp_config):
        import hfl.cli.main as main

        seen = {}
        monkeypatch.setattr(launcher.shutil, "which", lambda tool: "/usr/bin/claude")
        monkeypatch.setattr(main, "_local_or_pulled", lambda *a: MagicMock(name="m"))
        monkeypatch.setattr(launcher, "run", lambda *a, **k: seen.update(k) or 0)
        result = cli("claude", "-m", "m", "--", "-p", "fix it", "--allowedTools", "Read")
        assert result.exit_code == 0
        assert seen["extra"] == ["-p", "fix it", "--allowedTools", "Read"]
