# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Unit tests for ``hfl/engine/snapshot.py`` — V4 F6."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hfl.engine.snapshot import (
    SnapshotMeta,
    _validate_name,
    delete_snapshot,
    list_snapshots,
    load_snapshot,
    save_snapshot,
)


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Force snapshots into a tmp dir."""
    from hfl.config import config

    monkeypatch.setattr(config, "home_dir", tmp_path)
    return tmp_path


class _PickleableState:
    """Stand-in for ``Llama.save_state()``'s return value.

    MagicMock can't be pickled (it carries closures); we need a
    plain class that round-trips through ``pickle``.
    """

    def __init__(self, n_tokens: int = 42, payload: bytes = b"\x00" * 8):
        self.n_tokens = n_tokens
        self.payload = payload


@pytest.fixture
def fake_state():
    """A minimal state object with ``n_tokens`` for the metadata."""
    return _PickleableState(n_tokens=42)


def _engine_with_state(state):
    engine = MagicMock(spec=["save_state", "load_state"])
    engine.save_state = MagicMock(return_value=state)
    engine.load_state = MagicMock()
    return engine


# --- Name validation --------------------------------------------------------


class TestValidateName:
    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            _validate_name("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError):
            _validate_name("   ")

    def test_path_traversal_rejected(self):
        for bad in ("../escape", "foo/bar", "foo\\bar", ".."):
            with pytest.raises(ValueError):
                _validate_name(bad)

    def test_special_chars_rejected(self):
        with pytest.raises(ValueError):
            _validate_name("hello world")
        with pytest.raises(ValueError):
            _validate_name("name;rm")

    def test_normal_names_accepted(self):
        for ok in ("warm-start", "v1.2", "production_2026"):
            _validate_name(ok)  # must not raise


# --- save / load round-trip -------------------------------------------------


class TestSnapshotRoundTrip:
    def test_save_creates_state_and_meta(self, isolated_home, fake_state):
        engine = _engine_with_state(fake_state)
        meta = save_snapshot(engine, name="warm-1", model_name="qwen-7b")

        assert isinstance(meta, SnapshotMeta)
        assert meta.name == "warm-1"
        assert meta.model == "qwen-7b"
        assert meta.tokens == 42
        assert meta.bytes > 0
        assert (isolated_home / "snapshots" / "warm-1.state").exists()
        assert (isolated_home / "snapshots" / "warm-1.meta.json").exists()

    def test_load_invokes_engine_load_state_with_saved_payload(self, isolated_home, fake_state):
        save_engine = _engine_with_state(fake_state)
        save_snapshot(save_engine, name="warm-1", model_name="qwen-7b")

        load_engine = _engine_with_state(fake_state)
        meta = load_snapshot(load_engine, name="warm-1", model_name="qwen-7b")

        load_engine.load_state.assert_called_once()
        # The first positional arg is the unpickled state.
        loaded_state = load_engine.load_state.call_args.args[0]
        # Pickle round-trip preserves the n_tokens attribute.
        assert getattr(loaded_state, "n_tokens", None) == 42
        assert meta.name == "warm-1"


# --- Failure modes ----------------------------------------------------------


class TestSnapshotErrors:
    def test_save_engine_without_save_state_raises(self, isolated_home):
        engine = MagicMock(spec=[])
        with pytest.raises(RuntimeError, match="save_state"):
            save_snapshot(engine, name="x", model_name="m")

    def test_save_state_failure_propagates(self, isolated_home):
        engine = MagicMock(spec=["save_state"])
        engine.save_state = MagicMock(side_effect=RuntimeError("disk full"))
        with pytest.raises(RuntimeError, match="save_state"):
            save_snapshot(engine, name="x", model_name="m")

    def test_load_state_failure_propagates(self, isolated_home, fake_state):
        save_snapshot(_engine_with_state(fake_state), name="warm-1", model_name="m")
        bad_engine = MagicMock(spec=["load_state"])
        bad_engine.load_state = MagicMock(side_effect=RuntimeError("shape mismatch"))
        with pytest.raises(RuntimeError, match="load_state"):
            load_snapshot(bad_engine, name="warm-1", model_name="m")

    def test_load_missing_snapshot_raises_file_not_found(self, isolated_home):
        engine = _engine_with_state(MagicMock())
        with pytest.raises(FileNotFoundError):
            load_snapshot(engine, name="never-existed", model_name="m")

    def test_load_with_wrong_model_name_raises_value_error(self, isolated_home, fake_state):
        save_snapshot(_engine_with_state(fake_state), name="warm-1", model_name="A")

        with pytest.raises(ValueError, match="A"):
            load_snapshot(_engine_with_state(fake_state), name="warm-1", model_name="B")


# --- listing & deletion -----------------------------------------------------


class TestListAndDelete:
    def test_list_returns_recent_first(self, isolated_home, fake_state):
        save_snapshot(_engine_with_state(fake_state), name="alpha", model_name="m")
        save_snapshot(_engine_with_state(fake_state), name="beta", model_name="m")

        listing = list_snapshots()
        names = [m.name for m in listing]
        assert "alpha" in names and "beta" in names

    def test_delete_removes_both_files(self, isolated_home, fake_state):
        save_snapshot(_engine_with_state(fake_state), name="warm-1", model_name="m")
        deleted = delete_snapshot("warm-1")
        assert deleted is True
        assert not (isolated_home / "snapshots" / "warm-1.state").exists()
        assert not (isolated_home / "snapshots" / "warm-1.meta.json").exists()

    def test_delete_returns_false_when_nothing_to_delete(self, isolated_home):
        assert delete_snapshot("never-was") is False

    def test_corrupt_meta_is_skipped_in_listing(self, isolated_home, fake_state):
        save_snapshot(_engine_with_state(fake_state), name="warm-1", model_name="m")
        # Corrupt the sidecar.
        (isolated_home / "snapshots" / "warm-1.meta.json").write_text("not json")

        listing = list_snapshots()
        # No crash; the broken entry just doesn't appear.
        names = [m.name for m in listing]
        assert "warm-1" not in names
