# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Write real GGUF files for tests: a v3 header with the given metadata and
no tensors, optionally padded to look large. HFL reads GGUF metadata with
its own header reader, so tests give it real files rather than a fake of
the optional ``gguf`` package."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

# GGUF value types used here.
_UINT32, _BOOL, _STRING, _UINT64 = 4, 7, 8, 10


def _string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _value(value: Any) -> bytes:
    if isinstance(value, bool):
        return struct.pack("<I", _BOOL) + struct.pack("<?", value)
    if isinstance(value, int):
        if 0 <= value < 2**32:
            return struct.pack("<I", _UINT32) + struct.pack("<I", value)
        return struct.pack("<I", _UINT64) + struct.pack("<Q", value)
    if isinstance(value, str):
        return struct.pack("<I", _STRING) + _string(value)
    raise TypeError(f"unsupported GGUF test value: {value!r}")


def write_gguf(path: Path | str, fields: dict[str, Any], size_bytes: int = 0) -> Path:
    """A GGUF at ``path`` with ``fields`` (None values are left out), padded
    with zeros to ``size_bytes`` when that is larger than the header."""
    kept = {k: v for k, v in fields.items() if v is not None}
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kept))
    for key, value in kept.items():
        blob += _string(key) + _value(value)
    if size_bytes > len(blob):
        blob += b"\0" * (size_bytes - len(blob))
    target = Path(path)
    target.write_bytes(blob)
    return target


def model_fields(
    arch: str | None = "llama",
    *,
    block_count: int | None = None,
    embedding_length: int | None = None,
    context_length: int | None = None,
    head_count: int | None = None,
    head_count_kv: int | None = None,
    chat_template: str | None = None,
    add_bos_token: bool | None = None,
) -> dict[str, Any]:
    """The metadata keys HFL reads, for an ``arch`` model."""
    if arch is None:
        return {}
    return {
        "general.architecture": arch,
        f"{arch}.block_count": block_count,
        f"{arch}.embedding_length": embedding_length,
        f"{arch}.context_length": context_length,
        f"{arch}.attention.head_count": head_count,
        f"{arch}.attention.head_count_kv": head_count_kv,
        "tokenizer.chat_template": chat_template,
        "tokenizer.ggml.add_bos_token": add_bos_token,
    }
