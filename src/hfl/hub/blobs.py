# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Content-addressed blob storage for ``POST /api/blobs/:digest``.

Ollama's create flow works in two steps: the client POSTs raw bytes
(e.g. a GGUF) to ``/api/blobs/sha256:<hex>`` first, then references
that digest from the Modelfile of a subsequent ``/api/create``. The
blob API is therefore the storage plumbing underneath ``/api/create``;
it is not useful on its own but must exist for the client contract
to round-trip.

Storage layout::

    <home_dir>/blobs/sha256-<hex>

The filename uses a dash rather than a colon because FAT32/NTFS do
not allow colons in filenames. The network form (in URLs and
Modelfiles) keeps the ``sha256:<hex>`` colon form — Ollama normalises
between the two the same way.

Path traversal is blocked at the door: the digest is validated to
match ``[a-fA-F0-9]{64}`` before it touches ``Path.joinpath``. A
failed validation surfaces as ``InvalidBlobDigestError`` which the
route maps to a 400.

Writes are atomic: bytes land in a temporary file, SHA-256 is
computed on the stream as it is read, and the temp file is renamed
into place only if the final digest matches the request path. If it
does not, the temp is deleted and the caller gets a 400.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import AsyncIterator

from hfl.config import config

__all__ = [
    "InvalidBlobDigestError",
    "DigestMismatchError",
    "blob_dir",
    "blob_path",
    "blob_exists",
    "parse_digest",
    "write_blob_stream",
    "DEFAULT_CHUNK_SIZE",
]

# 1 MiB default chunk size for streaming writes. Large enough that
# per-chunk syscall overhead is negligible on multi-GB GGUFs, small
# enough that the hasher's intermediate state stays cache-friendly.
DEFAULT_CHUNK_SIZE = 1 * 1024 * 1024

_DIGEST_RE = re.compile(r"^([a-fA-F0-9]{64})$")


class InvalidBlobDigestError(ValueError):
    """Raised when a digest string is malformed.

    The ``/api/blobs`` route maps this to HTTP 400. Only canonical
    ``sha256:<64 hex chars>`` (with or without the ``sha256:`` prefix)
    is accepted; any other shape triggers this error.
    """


class DigestMismatchError(ValueError):
    """Raised when a stream's computed SHA-256 differs from the request path.

    The blob route maps this to HTTP 400. The message contains both
    the expected and computed digests so the client can diagnose
    transit-layer corruption.
    """

    def __init__(self, expected: str, actual: str) -> None:
        super().__init__(
            f"Digest mismatch: path requested sha256:{expected[:12]}…, "
            f"computed sha256:{actual[:12]}…"
        )
        self.expected = expected
        self.actual = actual


def blob_dir() -> Path:
    """Return the directory that holds blob files, creating it if missing."""
    path = config.home_dir / "blobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_digest(digest: str) -> str:
    """Normalise ``digest`` to a bare lowercase 64-char hex string.

    Accepts ``sha256:<hex>``, ``sha256-<hex>``, or a bare hex string.
    Anything else raises ``InvalidBlobDigestError``. The returned
    value never contains the ``sha256`` algorithm prefix.
    """
    if not isinstance(digest, str) or not digest:
        raise InvalidBlobDigestError("digest must be a non-empty string")

    # Strip an ``sha256:`` or ``sha256-`` prefix if present.
    if digest.lower().startswith("sha256:") or digest.lower().startswith("sha256-"):
        digest = digest.split(":", 1)[-1] if ":" in digest else digest.split("-", 1)[-1]

    if not _DIGEST_RE.match(digest):
        raise InvalidBlobDigestError("digest must be 64 hexadecimal characters (sha-256)")
    return digest.lower()


def blob_path(digest: str) -> Path:
    """Resolve a validated digest to its on-disk ``Path``.

    Does not check existence. Call ``blob_exists`` for that.
    """
    hex_digest = parse_digest(digest)
    # Filename uses ``sha256-<hex>`` (dash) for Windows-safe storage.
    return blob_dir() / f"sha256-{hex_digest}"


def blob_exists(digest: str) -> bool:
    """Return True iff a blob with the given digest is stored locally."""
    try:
        return blob_path(digest).is_file()
    except InvalidBlobDigestError:
        return False


async def write_blob_stream(
    expected_digest: str,
    chunks: AsyncIterator[bytes],
    *,
    chunk_limit: int | None = None,
) -> int:
    """Stream ``chunks`` into the blob store, validating SHA-256.

    Writes go to a temp file in the same directory as the final blob
    (so the rename is atomic), with the hasher fed on every chunk.
    When the stream ends:

    - if the computed digest matches ``expected_digest``, the temp
      is renamed into place and the byte count is returned;
    - otherwise ``DigestMismatchError`` is raised and the temp is
      removed.

    ``chunk_limit`` is an optional per-blob byte cap. Passing
    ``None`` disables the cap (the default; GGUFs routinely exceed
    the global request-body limit).
    """
    expected = parse_digest(expected_digest)
    final = blob_path(expected)
    if final.exists():
        # Already present — drain the stream so the caller's
        # connection closes cleanly, then report success.
        received = 0
        async for chunk in chunks:
            received += len(chunk)
            if chunk_limit is not None and received > chunk_limit:
                raise InvalidBlobDigestError(
                    f"blob exceeds configured per-request limit ({chunk_limit} bytes)"
                )
        return final.stat().st_size

    hasher = hashlib.sha256()
    total = 0
    # ``delete=False`` because we rename on success; on failure the
    # ``finally`` branch unlinks explicitly.
    tmp = tempfile.NamedTemporaryFile(
        dir=str(blob_dir()),
        prefix=".tmp-sha256-",
        delete=False,
    )
    tmp_path = Path(tmp.name)
    try:
        async for chunk in chunks:
            if not chunk:
                continue
            total += len(chunk)
            if chunk_limit is not None and total > chunk_limit:
                raise InvalidBlobDigestError(
                    f"blob exceeds configured per-request limit ({chunk_limit} bytes)"
                )
            hasher.update(chunk)
            tmp.write(chunk)
        tmp.flush()
        os.fsync(tmp.fileno())
    except Exception:
        tmp.close()
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    finally:
        try:
            tmp.close()
        except Exception:
            pass

    actual = hasher.hexdigest()
    if actual != expected:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise DigestMismatchError(expected=expected, actual=actual)

    # Atomic rename — on POSIX this never leaves a half-written blob
    # visible under its final name. On Windows, ``os.replace`` has the
    # same semantics.
    os.replace(str(tmp_path), str(final))
    return total
