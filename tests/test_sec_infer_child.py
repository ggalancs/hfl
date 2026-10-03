# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Security: what llama-server and its guard are started with.

* Their environment was HFL's whole one: HF_TOKEN, HFL_API_KEY and the
  search providers' keys reached a process that parses whatever clients
  send. They are left out now; llama-server's own key is still given.
* The guard ran as ``python -m hfl.engine._child_guard``, which puts the
  working directory first on ``sys.path``: ``hfl serve`` started from a
  directory holding an ``hfl/`` package ran that package's code.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hfl.engine import llama_server
from hfl.utils import self_exec

SECRETS = {
    "HF_TOKEN": "hf_secret",
    "HUGGING_FACE_HUB_TOKEN": "hf_secret2",
    "HFL_API_KEY": "hfl-key",
    "TAVILY_API_KEY": "tvly",
    "BRAVE_API_KEY": "brave",
    "hfl_some_secret": "lower-case name",
}


def test_child_environment_has_no_secrets(monkeypatch):
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HFL_KV_CACHE_TYPE", "q8_0")
    monkeypatch.setenv("LLAMA_API_KEY", "stale")
    env = llama_server._child_env("fresh-key")
    assert not set(SECRETS) & set(env)
    assert env["LLAMA_API_KEY"] == "fresh-key"
    assert env["HFL_KV_CACHE_TYPE"] == "q8_0"  # not a secret: kept
    assert env.get("PATH") == os.environ.get("PATH")


def test_start_server_starts_the_guard_with_that_environment(monkeypatch, tmp_path):
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    seen: dict = {}

    class Exited:
        def poll(self):
            return 1

    def popen(argv, **kwargs):
        seen.update(kwargs)
        return Exited()

    monkeypatch.setattr(llama_server.subprocess, "Popen", popen)
    monkeypatch.setattr(llama_server, "stop_server", lambda proc: None)
    with pytest.raises(RuntimeError, match="exited"):
        llama_server.start_server(["llama-server"], "m.gguf", tmp_path / "log", 5)
    assert not set(SECRETS) & set(seen["env"])
    assert seen["env"]["LLAMA_API_KEY"]


@pytest.fixture
def planted(tmp_path) -> Path:
    """A directory holding an ``hfl`` package that announces itself."""
    for package in ("hfl", "hfl/engine", "hfl/cli"):
        (tmp_path / package).mkdir(exist_ok=True)
        (tmp_path / package / "__init__.py").write_text(
            "import sys\nprint('PLANTED')\nsys.exit(99)\n"
        )
    return tmp_path


def _run_from(directory: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    # The test process's own ``hfl`` (an editable install or ``src/``).
    src = str(Path(self_exec.__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH")) if p)
    env.pop("PYTHONSAFEPATH", None)
    return subprocess.run(argv, capture_output=True, text=True, timeout=60, cwd=directory, env=env)


def test_guard_started_in_a_planted_directory_runs_hfl(monkeypatch, planted):
    monkeypatch.setattr(self_exec, "is_frozen", lambda: False)
    argv = self_exec.child_guard_argv(os.getpid(), [sys.executable, "-c", "print('started')"])
    out = _run_from(planted, argv)
    assert "PLANTED" not in out.stdout
    assert out.returncode == 0 and "started" in out.stdout, out.stderr[-500:]


def test_hfl_started_in_a_planted_directory_runs_hfl(monkeypatch, planted):
    monkeypatch.setattr(self_exec, "is_frozen", lambda: False)
    out = _run_from(planted, self_exec.hfl_argv("--version"))
    assert "PLANTED" not in out.stdout
    assert out.returncode == 0, out.stderr[-500:]


def test_an_executable_still_runs_itself(monkeypatch):
    monkeypatch.setattr(self_exec, "is_frozen", lambda: True)
    assert self_exec.child_guard_argv(42, ["x"]) == [
        sys.executable, self_exec.CHILD_GUARD_FLAG, "42", "--", "x",
    ]  # fmt: skip
