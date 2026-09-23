# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Saved chat sessions, reachable at last.

`core/sessions.py` shipped a complete, tested store — save, load, list,
delete, name validation, a human-readable JSON layout chosen so people
could edit sessions by hand. Nothing imported it. A user could not save a
session, resume one, or discover the feature existed.

Wiring it up is `hfl run --session <name>` plus `hfl sessions
list|show|rm`. The design decision worth pinning is *when* it writes:
after every exchange, not once at exit. The restarts worth surviving are
the unplanned ones — a crash, a Ctrl-C, an OOM kill — and saving only on
a clean exit would lose exactly the sessions the user wanted back.

The store itself is covered by test_sessions.py. These tests are about
the connection: the CLI reaching the store, and the store surviving the
ways a chat really ends.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from hfl.cli.main import app

runner = CliRunner()


@pytest.fixture
def session_home(tmp_path, monkeypatch):
    """Keep every test out of the developer's real ~/.hfl/sessions.

    Redirects ``config.home_dir`` rather than replacing ``sessions_dir``:
    that function creates the directory on the way past, and stubbing it
    out with a plain path meant the first write failed on a missing
    parent — a fixture inventing a bug the code does not have.
    """
    import hfl.config as hfl_config

    monkeypatch.setattr(hfl_config.config, "home_dir", tmp_path / ".hfl")
    return tmp_path / ".hfl" / "sessions"


def _write(session_home, name, messages, model="qwen"):
    session_home.mkdir(parents=True, exist_ok=True)
    (session_home / f"{name}.json").write_text(
        json.dumps(
            {
                "name": name,
                "model": model,
                "created_at": "2026-09-23T10:00:00+00:00",
                "updated_at": "2026-09-23T10:00:00+00:00",
                "options": {},
                "messages": messages,
                "system": None,
            }
        ),
        encoding="utf-8",
    )


class TestTheCommandGroupExists:
    """Before this change there was no way to reach the store at all."""

    def test_sessions_list_is_a_command(self):
        result = runner.invoke(app, ["sessions", "--help"])
        assert result.exit_code == 0
        for verb in ("list", "show", "rm"):
            assert verb in result.stdout

    def test_run_accepts_a_session_name(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--session" in result.stdout


class TestListing:
    def test_empty_says_how_to_start_one(self, session_home):
        result = runner.invoke(app, ["sessions", "list"])
        assert result.exit_code == 0
        assert "--session" in result.stdout, (
            "an empty state that does not say how to leave it is a dead end"
        )

    def test_shows_name_model_and_message_count(self, session_home):
        _write(session_home, "alpha", [{"role": "user", "content": "hi"}], model="qwen-7b")
        result = runner.invoke(app, ["sessions", "list"])
        assert result.exit_code == 0
        assert "alpha" in result.stdout
        assert "qwen-7b" in result.stdout


class TestShowing:
    def test_prints_the_conversation(self, session_home):
        _write(
            session_home,
            "beta",
            [
                {"role": "user", "content": "what is 2+2"},
                {"role": "assistant", "content": "four"},
            ],
        )
        result = runner.invoke(app, ["sessions", "show", "beta"])
        assert result.exit_code == 0
        assert "what is 2+2" in result.stdout
        assert "four" in result.stdout

    def test_unknown_name_fails_and_says_how_to_look(self, session_home):
        result = runner.invoke(app, ["sessions", "show", "nope"])
        assert result.exit_code == 1
        assert "sessions list" in result.stdout


class TestDeleting:
    def test_removes_the_file(self, session_home):
        _write(session_home, "gamma", [])
        assert (session_home / "gamma.json").exists()
        result = runner.invoke(app, ["sessions", "rm", "gamma"])
        assert result.exit_code == 0
        assert not (session_home / "gamma.json").exists()

    def test_deleting_what_is_not_there_is_an_error_not_a_shrug(self, session_home):
        result = runner.invoke(app, ["sessions", "rm", "ghost"])
        assert result.exit_code == 1


class TestPersistenceHappensEveryTurn:
    """The decision this feature turns on."""

    def test_run_saves_after_each_exchange_not_only_at_exit(self):
        """Asserted over the AST, not over string offsets.

        A first version searched for `_persist()` after the assistant
        append and compared its offset to `engine.unload()`. Deleting the
        in-loop call left the exit call, which still sits before
        `unload()`, so the check passed on the broken code. Locating the
        `while` node and asking whether a call lives inside it is the
        question that was actually meant.
        """
        import ast
        import inspect
        import textwrap

        from hfl.cli import main

        tree = ast.parse(textwrap.dedent(inspect.getsource(main.run)))
        loops = [n for n in ast.walk(tree) if isinstance(n, ast.While)]
        assert loops, "the chat loop disappeared"

        def calls_persist(node) -> bool:
            return any(
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "_persist"
                for sub in ast.walk(node)
            )

        assert any(calls_persist(loop) for loop in loops), (
            "no _persist() call inside the chat loop, so the session is only "
            "written when the user exits cleanly — a crash or Ctrl-C loses the "
            "conversation, which is the case the feature exists for"
        )

    def test_a_resumed_session_keeps_its_earlier_messages(self, session_home):
        """The round trip, through the real store."""
        from hfl.core.sessions import ChatSession, load_session, save_session

        first = ChatSession(name="delta", model="qwen")
        first.messages = [{"role": "user", "content": "one"}]
        save_session(first)

        reopened = load_session("delta")
        reopened.messages.append({"role": "assistant", "content": "two"})
        save_session(reopened)

        assert [m["content"] for m in load_session("delta").messages] == ["one", "two"]

    def test_the_system_prompt_is_not_duplicated_on_resume(self, session_home):
        """`--system` plus a resumed session that already has one.

        Appending a second system message would silently change the
        model's instructions on every resume.
        """
        import inspect

        from hfl.cli import main

        source = inspect.getsource(main.run)
        assert 'not any(m.role == "system" for m in messages)' in source, (
            "run() appends the --system prompt unconditionally, so resuming a "
            "session that already carries one stacks them"
        )
