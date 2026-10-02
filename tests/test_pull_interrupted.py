# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A download that a full disk or a lost Hub interrupts ends with a message.

Measured (audit G2, G3): the pull stopped with a traceback — hf_xet's
``RuntimeError: … No space left on device (os error 28)`` and
huggingface_hub's ``LocalEntryNotFoundError`` after its retries. These pin
the messages, and that anything else still surfaces as it is.
"""

from __future__ import annotations

import errno
from types import SimpleNamespace

import pytest
import typer


def _pull_failing_with(monkeypatch, exc: BaseException):
    from hfl.cli import main

    def pull_model(resolved):
        raise exc

    monkeypatch.setattr("hfl.hub.downloader.pull_model", pull_model)
    return main._download_or_exit


XET_FULL = RuntimeError(
    "Task error: File reconstruction error: IO Error: No space left on device (os error 28)"
)


@pytest.mark.parametrize(
    "exc",
    [XET_FULL, OSError(errno.ENOSPC, "No space left on device")],
    ids=["hf_xet", "oserror"],
)
def test_a_full_disk_says_so(monkeypatch, capsys, exc):
    download = _pull_failing_with(monkeypatch, exc)
    with pytest.raises(typer.Exit) as done:
        download(SimpleNamespace(repo_id="o/m"))
    assert done.value.exit_code == 1
    said = capsys.readouterr().out
    assert "disk filled up" in said and "Traceback" not in said


def test_a_full_disk_found_deeper_in_the_chain(monkeypatch, capsys):
    try:
        try:
            raise OSError(errno.ENOSPC, "No space left on device")
        except OSError as inner:
            raise RuntimeError("download failed") from inner
    except RuntimeError as outer:
        exc = outer
    download = _pull_failing_with(monkeypatch, exc)
    with pytest.raises(typer.Exit):
        download(SimpleNamespace(repo_id="o/m"))
    assert "disk filled up" in capsys.readouterr().out


def test_a_lost_hub_says_so(monkeypatch, capsys):
    from huggingface_hub.errors import LocalEntryNotFoundError

    download = _pull_failing_with(monkeypatch, LocalEntryNotFoundError("cannot find"))
    with pytest.raises(typer.Exit) as done:
        download(SimpleNamespace(repo_id="o/m"))
    assert done.value.exit_code == 1
    assert "connection to huggingface.co was lost" in capsys.readouterr().out


def test_anything_else_is_not_disguised(monkeypatch):
    download = _pull_failing_with(monkeypatch, ValueError("a bug"))
    with pytest.raises(ValueError, match="a bug"):
        download(SimpleNamespace(repo_id="o/m"))
