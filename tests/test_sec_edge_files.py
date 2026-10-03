# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Private files stay private; the HF token prompt never echoes and never
runs where nobody can answer it."""

from __future__ import annotations

import os
import stat
import threading
from unittest.mock import MagicMock, patch

import pytest

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


# ----------------------------------------------------------------------
# Finding 6: transcripts, sessions and logs are owner-only
# ----------------------------------------------------------------------


@posix_only
def test_a_saved_session_is_owner_only(temp_config):
    from hfl.core import sessions

    old = os.umask(0o022)
    try:
        path = sessions.save_session(sessions.ChatSession(name="s", model="m"))
    finally:
        os.umask(old)
    assert _mode(path) == 0o600
    assert _mode(path.parent) == 0o700


@posix_only
def test_an_old_world_readable_session_is_tightened_on_save(temp_config):
    from hfl.core import sessions

    sdir = temp_config.home_dir / "sessions"
    sdir.chmod(0o755)
    stale_tmp = sdir / "s.json.tmp"  # left by an interrupted write
    stale_tmp.write_text("{}")
    stale_tmp.chmod(0o644)
    path = sessions.save_session(sessions.ChatSession(name="s", model="m"))
    assert _mode(path) == 0o600
    assert _mode(sdir) == 0o700


@posix_only
def test_a_new_home_is_private_and_an_existing_one_is_left_alone(tmp_path):
    from hfl.config import HFLConfig

    old = os.umask(0o022)
    try:
        fresh = tmp_path / "fresh"
        HFLConfig(home_dir=fresh).ensure_dirs()
        assert _mode(fresh) == 0o700
        assert _mode(fresh / "sessions") == 0o700
        assert _mode(fresh / "logs") == 0o700

        chosen = tmp_path / "chosen"
        chosen.mkdir(mode=0o755)
        chosen.chmod(0o755)
        (chosen / "logs").mkdir(mode=0o755)
        (chosen / "logs").chmod(0o755)
        HFLConfig(home_dir=chosen).ensure_dirs()
        assert _mode(chosen) == 0o755  # the user's directory, as found
        assert _mode(chosen / "logs") == 0o700  # HFL's own, tightened
    finally:
        os.umask(old)


# ----------------------------------------------------------------------
# Finding 9: the HF token prompt
# ----------------------------------------------------------------------


def _gated_api():
    from huggingface_hub.utils import HfHubHTTPError

    response = MagicMock(status_code=401, headers={})
    api = MagicMock()
    api.model_info.side_effect = [HfHubHTTPError("401", response=response), MagicMock()]
    return api


def test_no_prompt_without_a_terminal():
    from hfl.hub import auth

    prompts: list = []
    with (
        patch.object(auth, "HfApi", return_value=_gated_api()),
        patch.object(auth, "get_hf_token", return_value=None),
        patch.object(auth, "stdin_is_terminal", return_value=False),
        patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **k: prompts.append(k) or "t"),
    ):
        with pytest.raises(RuntimeError, match="HF_TOKEN"):
            auth.ensure_auth("org/gated")
    assert prompts == []


def test_no_prompt_from_a_server_worker_thread():
    """/api/pull runs the pull in a worker thread of `hfl serve`."""
    from hfl.hub import auth

    prompts: list = []
    errors: list = []

    def _pull():
        try:
            auth.ensure_auth("org/gated")
        except RuntimeError as exc:
            errors.append(exc)

    with (
        patch.object(auth, "HfApi", return_value=_gated_api()),
        patch.object(auth, "get_hf_token", return_value=None),
        patch.object(auth, "stdin_is_terminal", return_value=True),
        patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **k: prompts.append(k) or "t"),
    ):
        worker = threading.Thread(target=_pull)
        worker.start()
        worker.join(10)
    assert prompts == [] and len(errors) == 1


def test_the_cli_prompt_does_not_echo():
    from hfl.hub import auth

    prompts: list = []
    with (
        patch.object(auth, "HfApi", return_value=_gated_api()),
        patch.object(auth, "get_hf_token", return_value=None),
        patch.object(auth, "stdin_is_terminal", return_value=True),
        patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **k: prompts.append(k) or "t"),
    ):
        assert auth.ensure_auth("org/gated") == "t"
    assert prompts == [{"password": True}]
