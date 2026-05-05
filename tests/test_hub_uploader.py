# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Unit tests for ``hfl/hub/uploader.py`` helpers.

Complements the ``/api/push`` integration tests by pinning the
behaviour of ``build_upload_plan`` and ``_iter_uploadable_files`` in
isolation — single-file vs directory paths, filtering rules, and the
two failure modes (bad ``target_repo_id`` shape, missing
``local_path``).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hfl.hub.uploader import _iter_uploadable_files, build_upload_plan, stream_push


@pytest.fixture
def manifest_factory():
    """Build a ``ModelManifest`` whose ``local_path`` points wherever
    the test wants. Lets each case exercise either a single file or a
    full directory snapshot without sharing fixture state."""
    from hfl.models.manifest import ModelManifest

    def _make(local_path: Path | str) -> ModelManifest:
        return ModelManifest(
            name="upload-test",
            repo_id="any/source",
            local_path=str(local_path),
            format="gguf",
            architecture="qwen",
            parameters="7B",
        )

    return _make


# --- _iter_uploadable_files ---------------------------------------------------


class TestIterUploadableFiles:
    def test_skips_pycache_dirs(self, tmp_path):
        (tmp_path / "model.gguf").write_bytes(b"x")
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "junk.py").write_text("dont_ship_me")

        files = sorted(p.name for p in _iter_uploadable_files(tmp_path))
        assert files == ["model.gguf"]

    def test_skips_pyc_pyo_files(self, tmp_path):
        (tmp_path / "model.gguf").write_bytes(b"x")
        (tmp_path / "stale.pyc").write_bytes(b"x")
        (tmp_path / "stale.pyo").write_bytes(b"x")

        files = sorted(p.name for p in _iter_uploadable_files(tmp_path))
        assert files == ["model.gguf"]

    def test_skips_dotfiles(self, tmp_path):
        (tmp_path / "model.gguf").write_bytes(b"x")
        (tmp_path / ".DS_Store").write_text("")
        (tmp_path / ".gitignore").write_text("*.pyc")

        files = sorted(p.name for p in _iter_uploadable_files(tmp_path))
        assert files == ["model.gguf"]

    def test_skips_lock_and_tmp_suffixes(self, tmp_path):
        (tmp_path / "model.gguf").write_bytes(b"x")
        (tmp_path / "ongoing.tmp").write_text("")
        (tmp_path / "ongoing.lock").write_text("")

        files = sorted(p.name for p in _iter_uploadable_files(tmp_path))
        assert files == ["model.gguf"]

    def test_includes_nested_legitimate_files(self, tmp_path):
        (tmp_path / "model.gguf").write_bytes(b"x")
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "tokenizer.json").write_text("{}")
        (sub / "vocab.txt").write_text("a")

        files = sorted(str(p.relative_to(tmp_path)) for p in _iter_uploadable_files(tmp_path))
        assert files == ["model.gguf", "subdir/tokenizer.json", "subdir/vocab.txt"]

    def test_returns_deterministic_order(self, tmp_path):
        # Reverse-create on purpose so naive directory order would
        # disagree with sorted order on some filesystems.
        for name in ("z.gguf", "m.gguf", "a.gguf"):
            (tmp_path / name).write_bytes(b"x")

        first_pass = [p.name for p in _iter_uploadable_files(tmp_path)]
        second_pass = [p.name for p in _iter_uploadable_files(tmp_path)]
        assert first_pass == second_pass
        assert first_pass == sorted(first_pass)


# --- build_upload_plan --------------------------------------------------------


class TestBuildUploadPlan:
    def test_single_file_path_uses_parent_as_local_dir(self, tmp_path, manifest_factory):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 16)

        plan = build_upload_plan(
            manifest_factory(gguf),
            target_repo_id="user/repo",
        )
        # The plan ships exactly the one file.
        assert plan.files == (gguf,)
        # ``local_dir`` is the parent so ``upload_folder`` walks the
        # right tree.
        assert plan.local_dir == tmp_path
        # Total bytes match the file on disk.
        assert plan.total_bytes == gguf.stat().st_size

    def test_directory_path_walks_uploadable_files(self, tmp_path, manifest_factory):
        (tmp_path / "model.safetensors").write_bytes(b"x" * 1024)
        (tmp_path / "tokenizer.json").write_text('{"vocab": {}}')
        # Junk that must not ship.
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "trash.py").write_text("noop")

        plan = build_upload_plan(
            manifest_factory(tmp_path),
            target_repo_id="user/repo",
        )
        names = sorted(p.name for p in plan.files)
        assert names == ["model.safetensors", "tokenizer.json"]
        assert plan.local_dir == tmp_path

    def test_target_repo_must_be_namespace_slash_model(self, tmp_path, manifest_factory):
        (tmp_path / "model.gguf").write_bytes(b"x")
        m = manifest_factory(tmp_path)

        for bad in ("no-namespace", "/leading-slash", ""):
            with pytest.raises(ValueError):
                build_upload_plan(m, target_repo_id=bad)

    def test_local_path_missing_raises(self, tmp_path, manifest_factory):
        ghost = tmp_path / "does-not-exist"
        with pytest.raises(FileNotFoundError):
            build_upload_plan(manifest_factory(ghost), target_repo_id="user/repo")

    def test_empty_directory_raises(self, tmp_path, manifest_factory):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            build_upload_plan(manifest_factory(empty), target_repo_id="user/repo")

    def test_plan_passes_through_revision(self, tmp_path, manifest_factory):
        (tmp_path / "model.gguf").write_bytes(b"x")
        plan = build_upload_plan(
            manifest_factory(tmp_path),
            target_repo_id="user/repo",
            revision="experimental",
        )
        assert plan.revision == "experimental"


# --- stream_push --------------------------------------------------------------


class TestStreamPushFailureModes:
    """``stream_push`` must convert both HF SDK failures into a
    ``response.failed`` event rather than letting the exception
    propagate — the route layer would otherwise emit a half-streamed
    NDJSON body and 500 the request."""

    @pytest.mark.asyncio
    async def test_create_repo_failure_emits_failed_event(self, tmp_path, manifest_factory):
        (tmp_path / "model.gguf").write_bytes(b"x")
        plan = build_upload_plan(manifest_factory(tmp_path), target_repo_id="user/repo")

        api = MagicMock()
        api.create_repo = MagicMock(side_effect=RuntimeError("rate limited"))
        api.upload_folder = MagicMock()  # must NOT be called

        events = [event async for event in stream_push(plan, api=api)]
        assert events[-1]["status"] == "failed"
        assert "create_repo" in events[-1]["error"]
        api.upload_folder.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_folder_failure_emits_failed_event(self, tmp_path, manifest_factory):
        (tmp_path / "model.gguf").write_bytes(b"x")
        plan = build_upload_plan(manifest_factory(tmp_path), target_repo_id="user/repo")

        api = MagicMock()
        api.create_repo = MagicMock(return_value=None)
        api.upload_folder = MagicMock(side_effect=RuntimeError("disk full"))

        events = [event async for event in stream_push(plan, api=api)]
        assert events[-1]["status"] == "failed"
        assert "upload_folder" in events[-1]["error"]

    @pytest.mark.asyncio
    async def test_success_path_includes_commit_url_when_available(
        self, tmp_path, manifest_factory
    ):
        (tmp_path / "model.gguf").write_bytes(b"x")
        plan = build_upload_plan(manifest_factory(tmp_path), target_repo_id="user/repo")

        api = MagicMock()
        api.create_repo = MagicMock(return_value=None)
        commit = MagicMock()
        commit.commit_url = "https://huggingface.co/user/repo/commit/abc"
        api.upload_folder = MagicMock(return_value=commit)

        events = [event async for event in stream_push(plan, api=api)]
        assert events[-1]["status"] == "success"
        assert events[-1]["commit_url"].endswith("/commit/abc")
        assert events[-1]["revision"] == "main"  # default when not set
