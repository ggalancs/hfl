# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Downloaded weights are compared with the sha256 the Hub publishes.
huggingface_hub checks the size only: a file damaged on disk was accepted
as it was (checked for real: one flipped byte, pulled again, kept)."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from hfl.exceptions import DownloadIntegrityError
from hfl.hub import downloader

GOOD = b"the real weights"
BAD = b"the real weightz"


def _setup(tmp_path, monkeypatch, *, hub_sha, redownload_gives):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "m.gguf").write_bytes(BAD)
    meta = folder / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (meta / "m.gguf.metadata").write_text("etag")
    monkeypatch.setattr(downloader, "_hub_sha256", lambda resolved, token: hub_sha)
    fetched: list[str] = []

    def fetch(repo_id, filename, revision, local_dir, token):
        fetched.append(filename)
        (local_dir / filename).write_bytes(redownload_gives)
        return local_dir / filename

    monkeypatch.setattr(downloader, "_download_file", fetch)
    resolved = SimpleNamespace(repo_id="o/r", revision=None)
    return folder, resolved, fetched


def test_a_damaged_file_is_downloaded_again(tmp_path, monkeypatch) -> None:
    good = hashlib.sha256(GOOD).hexdigest()
    folder, resolved, fetched = _setup(
        tmp_path, monkeypatch, hub_sha={"m.gguf": good}, redownload_gives=GOOD
    )
    downloader._verify_downloads(resolved, folder, None, ["m.gguf"])
    assert fetched == ["m.gguf"] and (folder / "m.gguf").read_bytes() == GOOD


def test_a_file_still_wrong_after_a_second_download_is_an_error(tmp_path, monkeypatch) -> None:
    good = hashlib.sha256(GOOD).hexdigest()
    folder, resolved, fetched = _setup(
        tmp_path, monkeypatch, hub_sha={"m.gguf": good}, redownload_gives=BAD
    )
    with pytest.raises(DownloadIntegrityError) as caught:
        downloader._verify_downloads(resolved, folder, None, ["m.gguf"])
    assert "m.gguf" in str(caught.value) and fetched == ["m.gguf"]
    assert not (folder / "m.gguf").exists()  # a wrong file is not left to be loaded


def test_a_matching_file_is_left_alone(tmp_path, monkeypatch) -> None:
    folder, resolved, fetched = _setup(
        tmp_path, monkeypatch, hub_sha={"m.gguf": hashlib.sha256(BAD).hexdigest()},
        redownload_gives=GOOD,
    )  # fmt: skip
    downloader._verify_downloads(resolved, folder, None, ["m.gguf"])
    assert fetched == []


def test_no_checksum_from_the_hub_is_not_checked_not_ok(tmp_path, monkeypatch, capsys) -> None:
    folder, resolved, fetched = _setup(tmp_path, monkeypatch, hub_sha={}, redownload_gives=GOOD)
    downloader._verify_downloads(resolved, folder, None, ["m.gguf"])
    assert fetched == [] and "not checked" in capsys.readouterr().out.lower()


def test_it_can_be_turned_off(tmp_path, monkeypatch) -> None:
    folder, resolved, fetched = _setup(tmp_path, monkeypatch, hub_sha=None, redownload_gives=GOOD)
    monkeypatch.setattr(downloader.config, "verify_downloads", False)
    downloader._verify_downloads(
        resolved, folder, None, ["m.gguf"]
    )  # no Hub call (None would fail)
    assert fetched == []
