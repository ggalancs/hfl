# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl launch`` — open a coding agent on a local model.

``hfl launch claude -m qwen-coder`` points Claude Code at HFL's
Anthropic Messages API; ``hfl launch codex -m qwen-coder`` points Codex at
its Responses API. Nothing is configured permanently: the tool runs as a
child process with the settings in its environment and command line.

A server already listening on the port is reused. Otherwise one is
started on 127.0.0.1 for the tool's lifetime and stopped when it exits,
whatever the exit. The model is loaded before the tool opens, so a model
that does not fit is reported here rather than as a failed first prompt,
and its real context window is passed on (Claude Code assumes 200K for a
model it does not know).
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from hfl.i18n import t

TOOLS = ("claude", "codex")
DEFAULT_TOKEN = "hfl"  # any non-empty token: HFL checks one only with --api-key
_LOOPBACK = ("127.0.0.1", "localhost", "::1")


@dataclass
class Launch:
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)


class LaunchError(Exception):
    """Something the user must fix; the message says what."""


def build_launch(
    tool: str,
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    context: int | None = None,
    extra: list[str] | None = None,
) -> Launch:
    """The command and environment that point ``tool`` at HFL."""
    extra = list(extra or [])
    if tool == "claude":
        env = {
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_AUTH_TOKEN": api_key or DEFAULT_TOKEN,
            # An ANTHROPIC_API_KEY in the user's shell would win over the
            # token and send the key to HFL; blank it for this process only.
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        }
        if context:
            env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(context)
        return Launch(argv=["claude", *extra], env=env)
    if tool == "codex":
        provider = f'name="HFL", base_url="{base_url}/v1", wire_api="responses"'
        env = {}
        if api_key:
            provider += ', env_key="HFL_API_KEY"'
            env["HFL_API_KEY"] = api_key
        argv = ["codex", "-c", "model_provider=hfl", "-c", f"model_providers.hfl={{{provider}}}"]
        if context:
            argv += ["-c", f"model_context_window={context}"]
        return Launch(argv=[*argv, "-m", model, *extra], env=env)
    raise LaunchError(t("commands.launch.messages.unknown_tool", tool=tool, tools=", ".join(TOOLS)))


def check_tool(tool: str) -> None:
    """Refuse an unknown or missing tool — before anything is downloaded."""
    if tool not in TOOLS:
        raise LaunchError(
            t("commands.launch.messages.unknown_tool", tool=tool, tools=", ".join(TOOLS))
        )
    if shutil.which(tool) is None:
        raise LaunchError(t("commands.launch.messages.not_installed", tool=tool))


def shell_lines(launch: Launch) -> list[str]:
    """``launch`` as lines to paste into a POSIX shell."""
    lines = [f"export {k}={shlex.quote(v)}" for k, v in launch.env.items()]
    return [*lines, shlex.join(launch.argv)]


def server_up(base_url: str) -> bool:
    import httpx

    try:
        return httpx.get(f"{base_url}/healthz", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


def start_server(port: int, api_key: str | None, log_path: Path) -> subprocess.Popen[bytes]:
    """``hfl serve`` on 127.0.0.1:``port``, logging to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-c",
        "from hfl.cli.main import app; app()",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if api_key:
        cmd += ["--api-key", api_key]
    with open(log_path, "ab") as log:
        return subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )


def wait_until_up(base_url: str, proc: subprocess.Popen[bytes], timeout: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        if server_up(base_url):
            return True
        time.sleep(0.5)
    return False


def stop_server(proc: subprocess.Popen[bytes]) -> None:
    """Stop the server this command started. Bounded: SIGTERM, then SIGKILL."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def preload(base_url: str, model: str, api_key: str | None) -> int | None:
    """Load ``model`` on the server; return its context window if reported.

    Raises :class:`LaunchError` with the server's own words when the model
    cannot be loaded (not found, does not fit in memory...).
    """
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = httpx.post(
            f"{base_url}/api/generate", json={"model": model}, headers=headers, timeout=900.0
        )
    except httpx.HTTPError as exc:
        raise LaunchError(t("commands.launch.messages.unreachable", url=base_url)) from exc
    if response.status_code != 200:
        raise LaunchError(_server_error(response))
    try:
        running = httpx.get(f"{base_url}/api/ps", headers=headers, timeout=10.0).json()
    except (httpx.HTTPError, ValueError):
        return None
    for entry in running.get("models", []):
        if model in (entry.get("name"), entry.get("model")):
            ctx = (entry.get("details") or {}).get("context_size")
            return ctx if isinstance(ctx, int) and ctx > 0 else None
    return None


def _server_error(response: object) -> str:
    status = getattr(response, "status_code", "?")
    try:
        body = response.json()  # type: ignore[attr-defined]
    except ValueError:
        return f"HTTP {status}"
    detail = body.get("error") or body.get("detail") or body
    if isinstance(detail, dict):
        detail = detail.get("error") or detail.get("message") or detail
    return f"HTTP {status}: {detail}"


def run(
    tool: str,
    model: str,
    *,
    host: str,
    port: int,
    api_key: str | None,
    extra: list[str],
    log_path: Path,
    say: Callable[[str], None],
) -> int:
    """Open ``tool`` on ``model``; return the tool's exit code."""
    check_tool(tool)
    base_url = f"http://{host}:{port}"
    started: subprocess.Popen[bytes] | None = None
    try:
        if not server_up(base_url):
            if host not in _LOOPBACK:
                raise LaunchError(t("commands.launch.messages.no_server", url=base_url))
            say(t("commands.launch.messages.starting", url=base_url, log=log_path))
            started = start_server(port, api_key, log_path)
            if not wait_until_up(base_url, started):
                raise LaunchError(t("commands.launch.messages.not_started", log=log_path))
        say(t("commands.launch.messages.loading", model=model))
        context = preload(base_url, model, api_key)
        launch = build_launch(tool, base_url, model, api_key=api_key, context=context, extra=extra)
        return subprocess.run(launch.argv, env={**os.environ, **launch.env}).returncode
    finally:
        if started is not None:
            stop_server(started)
