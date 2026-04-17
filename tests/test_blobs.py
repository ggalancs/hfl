# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the blob-storage helpers (Phase 6 P2-2).

Covers the storage layer in ``hfl.hub.blobs``. The HTTP routes in
``tests/test_routes_blobs.py`` exercise the same module through the
API surface; together they lock down the end-to-end contract for
``HEAD /api/blobs/:digest`` and ``POST /api/blobs/:digest``.
"""

from __future__ import annotations

import hashlib

import pytest

from hfl.hub.blobs import (
    DigestMismatchError,
    InvalidBlobDigestError,
    blob_dir,
    blob_exists,
    blob_path,
    parse_digest,
    write_blob_stream,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def _aiter(chunks):
    for chunk in chunks:
        yield chunk


# ----------------------------------------------------------------------
# parse_digest
# ----------------------------------------------------------------------


class TestParseDigest:
    def test_bare_hex_accepted(self):
        d = "a" * 64
        assert parse_digest(d) == d

    def test_sha256_prefix_stripped(self):
        d = "a" * 64
        assert parse_digest(f"sha256:{d}") == d

    def test_sha256_dash_prefix_stripped(self):
        d = "a" * 64
        assert parse_digest(f"sha256-{d}") == d

    def test_uppercase_is_lowered(self):
        d = "A" * 64
        assert parse_digest(d) == "a" * 64

    def test_rejects_short(self):
        with pytest.raises(InvalidBlobDigestError):
            parse_digest("abc")

    def test_rejects_non_hex(self):
        with pytest.raises(InvalidBlobDigestError):
            parse_digest("g" * 64)

    def test_rejects_empty(self):
        with pytest.raises(InvalidBlobDigestError):
            parse_digest("")

    def test_rejects_non_str(self):
        with pytest.raises(InvalidBlobDigestError):
            parse_digest(None)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# blob_dir / blob_path
# ----------------------------------------------------------------------


class TestBlobPathing:
    def test_blob_dir_is_created(self, temp_config):
        path = blob_dir()
        assert path.exists()
        assert path.is_dir()
        assert path.parent == temp_config.home_dir

    def test_blob_path_uses_dash_separator(self, temp_config):
        d = "b" * 64
        path = blob_path(d)
        assert path.name == f"sha256-{d}"

    def test_blob_path_rejects_traversal(self, temp_config):
        with pytest.raises(InvalidBlobDigestError):
            blob_path("../etc/passwd")

    def test_blob_exists_false_for_missing(self, temp_config):
        assert blob_exists("c" * 64) is False

    def test_blob_exists_false_for_malformed(self, temp_config):
        assert blob_exists("not-a-digest") is False


# ----------------------------------------------------------------------
# write_blob_stream
# ----------------------------------------------------------------------


class TestWriteBlobStream:
    async def test_happy_path_writes_file(self, temp_config):
        data = b"hello world"
        digest = _sha256(data)
        bytes_written = await write_blob_stream(digest, _aiter([data]))
        assert bytes_written == len(data)
        assert blob_exists(digest)
        assert blob_path(digest).read_bytes() == data

    async def test_multi_chunk_reassembles(self, temp_config):
        chunks = [b"part1-", b"part2-", b"part3"]
        data = b"".join(chunks)
        digest = _sha256(data)
        await write_blob_stream(digest, _aiter(chunks))
        assert blob_path(digest).read_bytes() == data

    async def test_digest_mismatch_raises_and_cleans_up(self, temp_config):
        data = b"expected payload"
        wrong_digest = "f" * 64
        with pytest.raises(DigestMismatchError) as exc_info:
            await write_blob_stream(wrong_digest, _aiter([data]))
        # Temp files under the blob dir must have been removed.
        leftovers = list(blob_dir().glob(".tmp-sha256-*"))
        assert leftovers == []
        # The final blob should NOT have been written.
        assert not blob_path(wrong_digest).exists()
        assert exc_info.value.expected == wrong_digest

    async def test_already_present_blob_is_accepted(self, temp_config):
        data = b"already there"
        digest = _sha256(data)
        # First write populates the store.
        await write_blob_stream(digest, _aiter([data]))
        # Second call drains the stream, returns size, makes no
        # modification.
        mtime_before = blob_path(digest).stat().st_mtime
        returned = await write_blob_stream(digest, _aiter([data]))
        assert returned == len(data)
        assert blob_path(digest).stat().st_mtime == mtime_before

    async def test_invalid_digest_rejected_up_front(self, temp_config):
        with pytest.raises(InvalidBlobDigestError):
            await write_blob_stream("nope", _aiter([b"data"]))

    async def test_chunk_limit_rejects_oversize(self, temp_config):
        data = b"x" * 1024
        digest = _sha256(data)
        with pytest.raises(InvalidBlobDigestError):
            await write_blob_stream(
                digest,
                _aiter([data]),
                chunk_limit=512,
            )
        # Temp cleaned up.
        leftovers = list(blob_dir().glob(".tmp-sha256-*"))
        assert leftovers == []

    async def test_empty_chunks_are_skipped(self, temp_config):
        data = b"abc"
        digest = _sha256(data)
        await write_blob_stream(
            digest,
            _aiter([b"", b"a", b"", b"bc", b""]),
        )
        assert blob_path(digest).read_bytes() == data

    async def test_atomic_rename_leaves_no_temp(self, temp_config):
        data = b"final bytes"
        digest = _sha256(data)
        await write_blob_stream(digest, _aiter([data]))
        temps = list(blob_dir().glob(".tmp-sha256-*"))
        assert temps == []
        assert blob_path(digest).is_file()
