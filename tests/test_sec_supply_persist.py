# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The provenance log is never lost to a crash or a bad file; snapshot
secrets and KV state are private from the moment they exist."""

from __future__ import annotations

import io
import json
import os
import stat
from pathlib import Path

import pytest

from hfl.models.provenance import ConversionRecord, ProvenanceLog


def _log(path: Path, n: int) -> None:
    log = ProvenanceLog(path)
    for i in range(n):
        log.record(ConversionRecord(source_repo=f"o/r{i}"))


def test_a_crash_mid_write_leaves_the_previous_log(tmp_path, monkeypatch):
    path = tmp_path / "provenance.json"
    _log(path, 3)
    before = path.read_text()
    real_open = io.open

    def crashing_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if "w" in mode:
            write = handle.write

            def half(data):
                write(data[: len(data) // 2])
                handle.flush()
                raise OSError("disk gone")

            handle.write = half
        return handle

    monkeypatch.setattr(io, "open", crashing_open)
    with pytest.raises(OSError):
        ProvenanceLog(path).record(ConversionRecord(source_repo="o/new"))
    monkeypatch.undo()
    assert path.read_text() == before
    assert len(json.loads(path.read_text())) == 3
    assert [p.name for p in tmp_path.iterdir()] == ["provenance.json"]  # no temp left


def test_an_unreadable_log_is_kept_not_overwritten(tmp_path):
    path = tmp_path / "provenance.json"
    damaged = '[{"source_repo": "o/r0"}, {"source_rep'
    path.write_text(damaged)
    ProvenanceLog(path).record(ConversionRecord(source_repo="o/new"))
    kept = [p for p in tmp_path.iterdir() if p.name.startswith("provenance.json.corrupt-")]
    assert len(kept) == 1 and kept[0].read_text() == damaged
    assert [r["source_repo"] for r in json.loads(path.read_text())] == ["o/new"]


@pytest.fixture
def _umask_022():
    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_the_snapshot_key_is_born_0600(temp_config, monkeypatch, _umask_022):
    from hfl.engine import snapshot

    modes: list[int] = []
    real_chmod = Path.chmod

    # Whatever the code does after creating it, the key must never have been
    # readable by others: record the mode the first time anything touches it.
    def no_chmod(self, mode, **kwargs):  # a chmod-after-create is the bug
        modes.append(stat.S_IMODE(self.stat().st_mode))
        return real_chmod(self, mode, **kwargs)

    monkeypatch.setattr(Path, "chmod", no_chmod)
    snapshot._snapshot_key()
    key = temp_config.home_dir / "snapshot.key"
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    assert all(m == 0o600 for m in modes), [oct(m) for m in modes]


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_an_existing_key_file_with_a_wide_mode_is_narrowed(temp_config, _umask_022):
    from hfl.engine import snapshot

    key = temp_config.home_dir / "snapshot.key"
    key.write_bytes(b"short")  # too short: regenerated in place
    key.chmod(0o644)
    snapshot._snapshot_key()
    assert stat.S_IMODE(key.stat().st_mode) == 0o600 and len(key.read_bytes()) == 32


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_a_state_file_is_0600(temp_config, _umask_022):
    from hfl.engine import snapshot

    class Engine:
        def save_state(self):
            return {"n_tokens": 3}

    snapshot.save_snapshot(Engine(), name="s1", model_name="m")
    state = temp_config.home_dir / "snapshots" / "s1.state"
    assert stat.S_IMODE(state.stat().st_mode) == 0o600
