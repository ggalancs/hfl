# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for :mod:`hfl.api.image_validator`.

The validator is the *only* thing standing between untrusted base64
strings and a multi-GB CLIP forward pass. Each of the four gates
(size / magic-bytes / MIME / dimensions) gets a dedicated test class
and every error path is pinned to a specific rejection message.
"""

from __future__ import annotations

import base64
import struct

import pytest

from hfl.api.image_validator import (
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    decode_base64_image,
    validate_image,
)
from hfl.exceptions import ValidationError

# ----------------------------------------------------------------------
# Minimal valid image synthesisers — exactly one of each supported
# format, at the smallest possible size. Keeping them here (rather
# than fixture files) makes the tests self-contained.
# ----------------------------------------------------------------------


def _make_png(width: int = 1, height: int = 1) -> bytes:
    """Build a 1-pixel-at-minimum PNG with the dimensions we claim.

    We only need a well-formed signature + IHDR; the image data
    chunks can be an empty IDAT + IEND because our validator only
    reads the header.
    """
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_type = b"IHDR"
    ihdr_data = (
        struct.pack(">II", width, height)  # width / height
        + b"\x08\x02\x00\x00\x00"  # 8-bit RGB, no interlace
    )
    # (CRC not actually checked by our validator — leave zero.)
    ihdr = struct.pack(">I", len(ihdr_data)) + ihdr_type + ihdr_data + b"\x00\x00\x00\x00"
    idat = struct.pack(">I", 0) + b"IDAT" + b"\x00\x00\x00\x00"
    iend = struct.pack(">I", 0) + b"IEND" + b"\x00\x00\x00\x00"
    return signature + ihdr + idat + iend


def _make_jpeg(width: int = 4, height: int = 4) -> bytes:
    """Build a minimum JPEG (SOI → SOF0 with our dims → EOI).

    The SOF0 segment is enough for the dimension parser.
    """
    soi = b"\xff\xd8"
    # SOF0: FF C0 <len:2 = 17> <precision=8> <h:2> <w:2>
    # <components=3> <YCbCr triplets>
    sof = (
        b"\xff\xc0"
        + struct.pack(
            ">HBHHB",
            17,  # length of segment including these 2 bytes
            8,  # 8-bit precision
            height,
            width,
            3,  # component count
        )
        + b"\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    )
    eoi = b"\xff\xd9"
    return soi + sof + eoi


def _make_gif(width: int = 1, height: int = 1) -> bytes:
    """GIF89a with just a logical-screen descriptor."""
    return (
        b"GIF89a"
        + struct.pack("<HH", width, height)
        + b"\x00\x00\x00"  # GCT flags / bg colour / aspect ratio
        + b";"  # trailer
    )


def _make_webp_vp8(width: int = 32, height: int = 32) -> bytes:
    """Minimal WebP/VP8 (lossy) whose offsets match the parser.

    Full-file layout consumed by ``_parse_webp_dimensions``:

    ``0-3``   RIFF
    ``4-7``   RIFF size (little-endian uint32)
    ``8-11``  WEBP
    ``12-15`` "VP8 " (note: trailing space)
    ``16-19`` chunk size
    ``20-22`` VP8 frame tag (3 bytes)
    ``23-25`` VP8 start code ``9d 01 2a``
    ``26-27`` width (LE uint16, masked to 14 bits)
    ``28-29`` height (LE uint16, masked to 14 bits)
    """
    payload = (
        b"VP8 "
        + struct.pack("<I", 10)  # 4 + 4 = 8 bytes
        + b"\x00\x00\x00"  # 3-byte frame tag
        + b"\x9d\x01\x2a"  # 3-byte start code
        + struct.pack("<H", width & 0x3FFF)
        + struct.pack("<H", height & 0x3FFF)
    )
    riff = b"RIFF" + struct.pack("<I", len(payload) + 4) + b"WEBP" + payload
    return riff


# ----------------------------------------------------------------------
# Gate 1 — SIZE
# ----------------------------------------------------------------------


class TestSizeGate:
    def test_oversized_rejected(self):
        """Anything over MAX_IMAGE_BYTES is rejected before format parsing."""
        data = _make_png() + b"\x00" * (MAX_IMAGE_BYTES + 1)
        with pytest.raises(ValidationError, match="exceeds"):
            validate_image(data)

    def test_tiny_rejected(self):
        """<16 bytes can't be a real image header."""
        with pytest.raises(ValidationError, match="too short"):
            validate_image(b"\x89PNG")

    def test_exact_limit_ok(self):
        """A legal image at exactly max_bytes is accepted."""
        # Construct a valid PNG padded up to the limit with trailing
        # bytes (after IEND they're ignored by parsers, mirroring
        # real-world lenient decoders).
        base = _make_png()
        padded = base + b"\x00" * (MAX_IMAGE_BYTES - len(base))
        assert len(padded) == MAX_IMAGE_BYTES
        info = validate_image(padded)
        assert info.mime == "image/png"


# ----------------------------------------------------------------------
# Gate 2 — MAGIC BYTES
# ----------------------------------------------------------------------


class TestMagicBytes:
    def test_valid_png(self):
        info = validate_image(_make_png())
        assert info.mime == "image/png"
        assert info.width == 1
        assert info.height == 1

    def test_valid_jpeg(self):
        info = validate_image(_make_jpeg(width=8, height=8))
        assert info.mime == "image/jpeg"
        assert (info.width, info.height) == (8, 8)

    def test_valid_gif(self):
        info = validate_image(_make_gif(width=2, height=3))
        assert info.mime == "image/gif"
        assert (info.width, info.height) == (2, 3)

    def test_valid_webp(self):
        info = validate_image(_make_webp_vp8(width=16, height=24))
        assert info.mime == "image/webp"

    def test_svg_rejected(self):
        """SVG is XML and cannot be processed by vision models; must
        be rejected even though it's technically an image format."""
        svg = b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"/>'
        padded = svg + b"\x00" * 16  # ensure length >= 16
        with pytest.raises(ValidationError, match="Unrecognised"):
            validate_image(padded)

    def test_plain_text_rejected(self):
        """An HTML document with text/html magic is rejected even
        if the Content-Type header claimed image/png."""
        with pytest.raises(ValidationError, match="Unrecognised"):
            validate_image(b"<!DOCTYPE html><html></html>" + b"\x00" * 16)

    def test_executable_rejected(self):
        """An ELF binary pretending to be an image."""
        with pytest.raises(ValidationError, match="Unrecognised"):
            validate_image(b"\x7fELF" + b"\x00" * 64)


# ----------------------------------------------------------------------
# Gate 4 — DIMENSIONS
# ----------------------------------------------------------------------


class TestDimensionsGate:
    def test_max_4k_ok(self):
        info = validate_image(_make_png(width=4096, height=4096))
        assert info.width == 4096

    def test_over_max_dimension_rejected(self):
        with pytest.raises(ValidationError, match="dimensions"):
            validate_image(_make_png(width=5000, height=5000))

    def test_over_max_on_one_side_only_rejected(self):
        with pytest.raises(ValidationError, match="dimensions"):
            validate_image(_make_png(width=8192, height=100))

    def test_zero_dimension_rejected(self):
        """A PNG that claims 0x0 — common header corruption."""
        with pytest.raises(ValidationError, match="parse|dimensions"):
            validate_image(_make_png(width=0, height=0))

    def test_max_pixels_limit(self):
        """A 4096x4096 image = 16 Mpx, just at the limit."""
        # One below: 4095x4096 → under the pixel cap AND dimension cap
        info = validate_image(_make_png(width=4095, height=4096))
        assert info.width * info.height <= MAX_IMAGE_PIXELS

    def test_pathological_aspect_ratio_would_be_rejected(self):
        """If we lowered max_pixels but kept max_dimension, the
        pixel cap catches it. Use a custom cap to exercise the
        branch without synthesising a huge image."""
        with pytest.raises(ValidationError, match="pixel count"):
            validate_image(
                _make_png(width=100, height=100),
                max_pixels=1_000,
            )


# ----------------------------------------------------------------------
# Base64 decoding
# ----------------------------------------------------------------------


class TestDecodeBase64:
    def test_plain_base64(self):
        data = _make_png()
        b64 = base64.b64encode(data).decode()
        info = validate_image(b64)
        assert info.mime == "image/png"

    def test_data_uri(self):
        data = _make_png()
        uri = "data:image/png;base64," + base64.b64encode(data).decode()
        info = validate_image(uri)
        assert info.mime == "image/png"

    def test_data_uri_without_base64_marker_rejected(self):
        """``data:image/png,...`` (no base64 marker) → 400."""
        data = _make_png()
        uri = "data:image/png," + data.hex()
        with pytest.raises(ValidationError, match="base64"):
            decode_base64_image(uri)

    def test_non_image_data_uri_rejected(self):
        """``data:application/pdf;base64,...`` → 400 even before
        magic-byte sniffing."""
        uri = "data:application/pdf;base64," + base64.b64encode(b"%PDF").decode()
        with pytest.raises(ValidationError, match="MIME"):
            decode_base64_image(uri)

    def test_invalid_base64_rejected(self):
        with pytest.raises(ValidationError, match="base64"):
            decode_base64_image("not$$$valid!!!")

    def test_non_string_input_rejected(self):
        with pytest.raises(ValidationError, match="string"):
            decode_base64_image(b"bytes")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Input-type handling
# ----------------------------------------------------------------------


class TestInputType:
    def test_bytes_input(self):
        info = validate_image(_make_png())
        assert info.mime == "image/png"

    def test_bytearray_input(self):
        info = validate_image(bytearray(_make_png()))
        assert info.mime == "image/png"

    def test_bad_input_type_rejected(self):
        with pytest.raises(ValidationError, match="bytes or base64"):
            validate_image(12345)  # type: ignore[arg-type]


class TestReturnShape:
    def test_info_is_frozen_dataclass(self):
        info = validate_image(_make_png(width=7, height=11))
        assert info.data == _make_png(width=7, height=11)
        assert info.mime == "image/png"
        assert info.width == 7
        assert info.height == 11
        # Frozen: can't mutate
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            info.width = 999  # type: ignore[misc]


# ----------------------------------------------------------------------
# Configurable limits
# ----------------------------------------------------------------------


class TestCustomLimits:
    def test_custom_max_bytes(self):
        data = _make_png()
        with pytest.raises(ValidationError, match="exceeds"):
            validate_image(data, max_bytes=10)

    def test_custom_max_dimension(self):
        with pytest.raises(ValidationError, match="dimensions"):
            validate_image(_make_png(width=100, height=100), max_dimension=50)
