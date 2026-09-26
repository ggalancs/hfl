# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Read a few metadata keys from a GGUF header, with no dependency.

The ``gguf`` package is an optional extra (``hfl[convert]``) and Homebrew's
HFL does not ship it, yet whether a GGUF is an embedding model has to be
known everywhere: every GGUF used to be taken for a chat model, so no GGUF
embedding model could ever be used for embeddings.
"""

from __future__ import annotations

import functools
import os
import struct
from pathlib import Path
from typing import Any, BinaryIO

# GGUF value types: (struct format, size) for the fixed-size ones.
_FIXED = {
    0: ("<B", 1),
    1: ("<b", 1),
    2: ("<H", 2),
    3: ("<h", 2),
    4: ("<I", 4),
    5: ("<i", 4),
    6: ("<f", 4),
    7: ("<?", 1),
    10: ("<Q", 8),
    11: ("<q", 8),
    12: ("<d", 8),
}
_STRING, _ARRAY = 8, 9

# Architectures that are encoders only: embedding models whatever their keys.
_ENCODERS = {
    "bert",
    "nomic-bert",
    "nomic-bert-moe",
    "jina-bert-v2",
    "jina-bert-v3",
    "modern-bert",
    "neo-bert",
    "t5encoder",
    "gemma-embedding",
}


def _read(fh: BinaryIO, size: int) -> bytes:
    data = fh.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF header")
    return data


def _string(fh: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", _read(fh, 8))
    return _read(fh, length).decode("utf-8", errors="replace")


def _value(fh: BinaryIO, kind: int, keep: bool) -> Any:
    if kind in _FIXED:
        fmt, size = _FIXED[kind]
        return struct.unpack(fmt, _read(fh, size))[0]
    if kind == _STRING:
        return _string(fh)
    if kind == _ARRAY:
        (item_kind,) = struct.unpack("<I", _read(fh, 4))
        (count,) = struct.unpack("<Q", _read(fh, 8))
        if item_kind in _FIXED and not keep:
            fh.seek(_FIXED[item_kind][1] * count, 1)  # skipped, not read
            return None
        items = [_value(fh, item_kind, keep) for _ in range(count)]
        return items if keep else None
    raise ValueError(f"unknown GGUF value type {kind}")


def read_fields(path: Path | str, wanted: set[str] | None = None) -> dict[str, Any]:
    """The header's key/value pairs (only ``wanted`` ones when given).

    Raises ``ValueError`` or ``OSError`` for a file that is not a readable
    GGUF (v2 or v3).
    """
    found: dict[str, Any] = {}
    with open(path, "rb") as fh:
        if _read(fh, 4) != b"GGUF":
            raise ValueError("not a GGUF file")
        (version,) = struct.unpack("<I", _read(fh, 4))
        if version < 2:
            raise ValueError(f"GGUF version {version} is not supported")
        _tensors, count = struct.unpack("<QQ", _read(fh, 16))
        for _ in range(count):
            key = _string(fh)
            (kind,) = struct.unpack("<I", _read(fh, 4))
            keep = wanted is None or key in wanted
            value = _value(fh, kind, keep)
            if keep:
                found[key] = value
                if wanted is not None and wanted <= found.keys():
                    break
    return found


def is_embedding_gguf(path: Path | str) -> bool:
    """True when the GGUF is an embedding model: an encoder architecture, or
    one that declares how to pool its outputs (Qwen3-Embedding and the like;
    chat models do not). False, not an error, when unreadable.

    Remembered per file version: a model load asks several times, and a key
    a chat model lacks is only known absent after reading its vocabulary."""
    try:
        stat = os.stat(path)
    except OSError:
        return False
    return _is_embedding(str(path), stat.st_mtime_ns, stat.st_size)


@functools.lru_cache(maxsize=256)
def _is_embedding(path: str, _mtime_ns: int, _size: int) -> bool:
    try:
        arch = read_fields(path, {"general.architecture"}).get("general.architecture")
        if not isinstance(arch, str):
            return False
        if arch in _ENCODERS:
            return True
        pooling = read_fields(path, {f"{arch}.pooling_type"}).get(f"{arch}.pooling_type")
    except (OSError, ValueError, struct.error):
        return False
    # 0 is "none": a model that declares it does not pool is not one.
    return isinstance(pooling, int) and pooling > 0
