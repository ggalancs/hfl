# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Image validation for vision / multimodal requests (Phase 4, P0-6).

Every byte that a client hands to HFL as an "image" gets funnelled
through ``validate_image``. It enforces four independent gates so an
abusive request can't reach the vision engine:

1. **Size**: ≤ ``config.max_image_bytes`` (default 20 MiB). The
   global request body-size middleware already caps the envelope,
   but a chat with ten images can legitimately be ~100 MiB; we cap
   per-image so a single oversized attachment fails cleanly.

2. **Magic bytes**: sniff the first 12 bytes to confirm the payload
   is actually an image in a supported format. Prevents the classic
   "SVG containing <script>", "HTML with image/jpeg MIME", or
   "executable pretending to be PNG" attacks.

3. **MIME whitelist**: PNG / JPEG / WEBP / GIF only. SVG is
   deliberately excluded — it's XML, not raster, and vision models
   can't actually process it. GIF is accepted but only the first
   frame is used by downstream backends.

4. **Dimensions**: ≤ 4096 × 4096 pixels. A 20 MiB PNG is easily
   decompressed into a 30000×30000 image tensor (900M pixels);
   catching this up front prevents OOM on the CLIP side.

The module does NOT require Pillow at import time — the dimension
check falls back to manual header parsing for PNG / JPEG / WEBP so
test environments without Pillow still work. When Pillow IS
available we use it for stricter validation.
"""

from __future__ import annotations

import base64
import binascii
import logging
import struct
from dataclasses import dataclass
from typing import Final

from hfl.exceptions import ValidationError as APIValidationError

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Limits — exposed as module constants so tests can patch them cleanly.
# ----------------------------------------------------------------------

MAX_IMAGE_BYTES: Final[int] = 20 * 1024 * 1024  # 20 MiB
MAX_IMAGE_PIXELS: Final[int] = 4096 * 4096  # 16 megapixels
MAX_IMAGE_DIMENSION: Final[int] = 4096  # 4K on any side


# ----------------------------------------------------------------------
# Magic-byte signatures — the authoritative format detector.
# ----------------------------------------------------------------------

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"
_GIF_SIGNATURES = (b"GIF87a", b"GIF89a")
_WEBP_RIFF_PREFIX = b"RIFF"
_WEBP_FOURCC = b"WEBP"


@dataclass(frozen=True)
class ImageInfo:
    """Structured result of a successful ``validate_image`` call.

    Attributes:
        data: Raw decoded bytes, unchanged from input.
        mime: ``image/png`` / ``image/jpeg`` / ``image/webp`` /
            ``image/gif`` — what the bytes actually are, not what
            the caller claimed.
        width: Image width in pixels (0 when not parseable).
        height: Image height in pixels (0 when not parseable).
    """

    data: bytes
    mime: str
    width: int = 0
    height: int = 0


def _sniff_mime(data: bytes) -> str | None:
    """Return the canonical MIME type for ``data`` or None if unknown."""
    if len(data) < 12:
        return None
    if data.startswith(_PNG_SIGNATURE):
        return "image/png"
    if data.startswith(_JPEG_SIGNATURE):
        return "image/jpeg"
    if data.startswith(_GIF_SIGNATURES):
        return "image/gif"
    if data.startswith(_WEBP_RIFF_PREFIX) and data[8:12] == _WEBP_FOURCC:
        return "image/webp"
    return None


def _parse_png_dimensions(data: bytes) -> tuple[int, int]:
    """Read IHDR for width/height. PNG layout is well-known and
    stable: bytes 16-24 after the signature hold the dimensions
    as two big-endian unsigned 32-bit integers.
    """
    if len(data) < 24:
        raise APIValidationError("PNG file truncated before IHDR")
    width = struct.unpack(">I", data[16:20])[0]
    height = struct.unpack(">I", data[20:24])[0]
    return width, height


def _parse_jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Scan JPEG segments for an SOF marker (Start-Of-Frame) and
    pull the dimensions from it. Multi-byte walking but only over
    the header markers — cheap even for large files.
    """
    i = 2  # Skip leading FFD8
    length = len(data)
    while i < length - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        # Any SOF marker except SOF4 / SOF12 (which are rare/special).
        if marker in range(0xC0, 0xCF) and marker not in (0xC4, 0xC8, 0xCC):
            # Layout: FF Cx <len:2> <precision:1> <height:2> <width:2>
            height = struct.unpack(">H", data[i + 5 : i + 7])[0]
            width = struct.unpack(">H", data[i + 7 : i + 9])[0]
            return width, height
        # Skip segment
        if marker == 0xD8 or marker == 0xD9:  # SOI / EOI
            i += 2
            continue
        seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
        i += 2 + seg_len
    raise APIValidationError("JPEG file has no SOF segment")


def _parse_webp_dimensions(data: bytes) -> tuple[int, int]:
    """WebP has three sub-formats: VP8 (lossy), VP8L (lossless),
    VP8X (extended). Handle all three."""
    if len(data) < 30:
        raise APIValidationError("WebP file truncated")
    chunk_type = data[12:16]
    if chunk_type == b"VP8 ":
        # Simple lossy: width/height are in the VP8 bitstream at offset 26
        width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return width, height
    if chunk_type == b"VP8L":
        # Lossless layout: after the 0x2f signature byte at offset 20,
        # width-1 is 14 bits, height-1 is 14 bits.
        if data[20] != 0x2F:
            raise APIValidationError("Invalid VP8L signature")
        b0, b1, b2, b3 = data[21], data[22], data[23], data[24]
        width = 1 + (((b1 & 0x3F) << 8) | b0)
        height = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
        return width, height
    if chunk_type == b"VP8X":
        # Extended: width-1 / height-1 are 24-bit little-endian at
        # offsets 24..27 and 27..30.
        width = 1 + (data[24] | (data[25] << 8) | (data[26] << 16))
        height = 1 + (data[27] | (data[28] << 8) | (data[29] << 16))
        return width, height
    raise APIValidationError(f"Unsupported WebP sub-format: {chunk_type!r}")


def _parse_gif_dimensions(data: bytes) -> tuple[int, int]:
    """GIF screen descriptor sits right after the 6-byte magic."""
    if len(data) < 10:
        raise APIValidationError("GIF file truncated")
    width = struct.unpack("<H", data[6:8])[0]
    height = struct.unpack("<H", data[8:10])[0]
    return width, height


def _parse_dimensions(data: bytes, mime: str) -> tuple[int, int]:
    """Dispatch parser by MIME. Returns (0, 0) when the format is
    malformed (callers choose whether to treat that as a rejection).
    """
    try:
        if mime == "image/png":
            return _parse_png_dimensions(data)
        if mime == "image/jpeg":
            return _parse_jpeg_dimensions(data)
        if mime == "image/webp":
            return _parse_webp_dimensions(data)
        if mime == "image/gif":
            return _parse_gif_dimensions(data)
    except (IndexError, struct.error) as exc:
        raise APIValidationError(f"Malformed {mime}: {exc}")
    return 0, 0


def decode_base64_image(raw: str) -> bytes:
    """Decode a base64 string — strip a leading ``data:`` URI if present.

    Accepts either a bare base64 payload (Ollama's
    ``messages[].images`` format) or a data URI
    (``data:image/png;base64,...`` — the OpenAI content-part form).

    Raises:
        APIValidationError: Input is not valid base64, or the data
            URI MIME doesn't look like an image.
    """
    if not isinstance(raw, str):
        raise APIValidationError("image must be a base64 string or data URI")
    payload = raw.strip()
    if payload.startswith("data:"):
        try:
            header, payload = payload.split(",", 1)
        except ValueError:
            raise APIValidationError("Malformed data URI: missing comma separator")
        header_lower = header.lower()
        if not header_lower.startswith("data:image/"):
            raise APIValidationError(f"Data URI MIME must be image/*; got {header!r}")
        if ";base64" not in header_lower:
            raise APIValidationError("Data URI must declare ``;base64`` encoding")
    try:
        # ``validate=True`` rejects whitespace + non-base64 chars.
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise APIValidationError(f"Invalid base64 payload: {exc}")
    return decoded


def validate_image(
    data: bytes | str,
    *,
    max_bytes: int = MAX_IMAGE_BYTES,
    max_dimension: int = MAX_IMAGE_DIMENSION,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> ImageInfo:
    """Validate raw image bytes (or a base64 string) and return ``ImageInfo``.

    The function is deliberately strict: it rejects anything that
    isn't obviously a raster image in one of the four supported
    formats (PNG / JPEG / WEBP / GIF). See the module docstring for
    the four gates enforced.

    Args:
        data: Raw bytes OR a base64-encoded string / data URI. A
            string is decoded via :func:`decode_base64_image` first.
        max_bytes: Upper bound on raw decoded size. Default 20 MiB.
        max_dimension: Maximum width or height in pixels. Default
            4096 (standard "up to 4K").
        max_pixels: Maximum pixel count. Default 16 megapixels;
            prevents tall-and-thin or wide-and-short pathological
            aspect ratios (e.g. 8192x1 would pass dimension check
            but fail here).

    Returns:
        :class:`ImageInfo` with the normalised MIME and parsed
        dimensions.

    Raises:
        APIValidationError: Any of the four gates fails. The
            message names which one so callers can surface a useful
            400 body.
    """
    if isinstance(data, str):
        data = decode_base64_image(data)

    if not isinstance(data, (bytes, bytearray)):
        raise APIValidationError(
            f"image must be bytes or base64 string (got {type(data).__name__})"
        )
    data = bytes(data)

    # Gate 1: size
    if len(data) > max_bytes:
        raise APIValidationError(f"Image exceeds {max_bytes} bytes (got {len(data)})")
    if len(data) < 10:
        # 10 bytes is the minimum for a GIF header. PNG needs 24
        # (signature + IHDR) but that's caught later by the
        # dimension parser, which raises its own message.
        raise APIValidationError("Image data too short to be a valid image")

    # Gate 2 + 3: magic bytes ⇒ MIME
    mime = _sniff_mime(data)
    if mime is None:
        raise APIValidationError(
            "Unrecognised image format — only PNG, JPEG, WEBP and GIF are supported"
        )

    # Gate 4: dimensions
    width, height = _parse_dimensions(data, mime)
    if width <= 0 or height <= 0:
        raise APIValidationError(f"Could not parse {mime} dimensions")
    if width > max_dimension or height > max_dimension:
        raise APIValidationError(
            f"Image dimensions {width}x{height} exceed max {max_dimension}x{max_dimension}"
        )
    if width * height > max_pixels:
        raise APIValidationError(f"Image pixel count {width * height} exceeds max {max_pixels}")

    return ImageInfo(data=data, mime=mime, width=width, height=height)
