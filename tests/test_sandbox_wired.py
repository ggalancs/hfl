# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Process hardening at startup, connected to the flag that asks for it.

`core/sandbox.py` documented `--sandbox <mode>` and `HFL_SANDBOX` in its
own docstring. Neither existed: `hfl serve` had no such option and
nothing read the variable, so an operator following the module's own
description got a server that silently ran unhardened.

Two properties matter more than the mechanism, and both are asserted:

* **Ordering.** A restriction applied after the server is accepting
  requests protects nothing. It has to run before `start_server`.
* **Never fatal.** `apply_sandbox` is written not to raise, and the flag
  must inherit that. Opt-in hardening that refuses to boot on an
  unsupported platform is a denial of service the operator did not ask
  for — so an unavailable mode warns and serves.

The enforcement itself (seccomp-bpf on Linux, the codesign-dependent
macOS path) is covered by test_sandbox.py. These are about the wiring.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
from typer.testing import CliRunner

from hfl.cli.main import app, serve
from hfl.core.sandbox import SUPPORTED_MODES, apply_sandbox

runner = CliRunner()


class TestTheFlagExists:
    def test_serve_offers_sandbox(self):
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--sandbox" in result.stdout

    def test_the_help_names_the_env_var(self):
        """The module promised HFL_SANDBOX; the flag has to admit it too."""
        result = runner.invoke(app, ["serve", "--help"])
        assert "HFL_SANDBOX" in result.stdout


class TestOrdering:
    """A sandbox applied after the first request protects nothing."""

    @staticmethod
    def _serve_tree() -> ast.Module:
        return ast.parse(textwrap.dedent(inspect.getsource(serve)))

    def test_apply_sandbox_runs_before_start_server(self):
        tree = self._serve_tree()
        positions: dict[str, int] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in ("apply_sandbox", "start_server"):
                positions.setdefault(name, node.lineno)

        assert "apply_sandbox" in positions, "serve() never applies the sandbox"
        assert "start_server" in positions
        assert positions["apply_sandbox"] < positions["start_server"], (
            "the sandbox is applied after the server starts serving, which "
            "hardens nothing that matters"
        )

    def test_the_env_var_is_consulted(self):
        """Asserted over the AST, because the string also sits in a comment.

        A first version grepped the source for "HFL_SANDBOX" and passed
        after the lookup was deleted — the explanatory comment above it
        still mentioned the variable. Anchoring a check in the prose that
        describes the code is how a guard survives the code it guards.
        """
        for node in ast.walk(self._serve_tree()):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "attr", None) != "get":
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and arg.value == "HFL_SANDBOX":
                    return
        raise AssertionError(
            "serve() never reads HFL_SANDBOX from the environment, so a "
            "container cannot enable hardening without changing its command line"
        )


class TestNeverFatal:
    """Hardening is opt-in; failing to harden must not stop the server."""

    @pytest.mark.parametrize("mode", SUPPORTED_MODES)
    def test_every_supported_mode_returns_instead_of_raising(self, mode):
        result = apply_sandbox(mode)
        assert result.mode in SUPPORTED_MODES

    def test_an_unknown_mode_is_ignored_not_fatal(self):
        result = apply_sandbox("not-a-mode")
        assert not result.applied
        assert result.mode == "none"
        assert "unknown" in (result.reason or "").lower()

    def test_none_and_unset_are_no_ops(self):
        assert apply_sandbox(None).mode == "none"
        assert apply_sandbox("none").mode == "none"

    def test_serve_does_not_exit_when_hardening_is_unavailable(self, monkeypatch):
        """The whole point: a request for an unsupported mode still serves."""
        from hfl.cli import main

        started: list[bool] = []
        monkeypatch.setattr(main, "t", lambda key, **kw: key, raising=False)

        def fake_start(**kwargs):
            started.append(True)

        monkeypatch.setattr("hfl.api.server.start_server", fake_start)

        result = runner.invoke(app, ["serve", "--sandbox", "seccomp", "--port", "0"])
        assert started, "the server never started"
        assert result.exit_code == 0, (
            "requesting a sandbox mode this platform cannot apply stopped the "
            "server from booting — hardening is opt-in, not a precondition"
        )


class TestTheOperatorIsTold:
    def test_a_requested_but_unapplied_sandbox_is_surfaced(self, monkeypatch):
        """Silence here would read as success.

        The reason it could not be applied is the actionable half: 'not
        Linux' and 'seccomp unavailable in this kernel' need different
        responses from the operator.
        """
        source = inspect.getsource(serve)
        assert "not applied" in source or "_sandbox_result.reason" in source, (
            "serve() applies the sandbox but never reports a failure, so an "
            "operator who asked for hardening cannot tell whether they got it"
        )
