# SPDX-License-Identifier: HRUL-1.0
"""Security edge cases for image_validator.

Complements ``test_image_validator.py`` (happy paths + main gates) by
driving the malformed/truncated/exotic-format branches that an abusive
client could hit: truncated headers, WebP VP8L/VP8X sub-formats, an
invalid VP8L signature, unsupported WebP chunks, a JPEG with no SOF
segment, and malformed data URIs. These are the paths where a parser
bug would turn untrusted input into a crash or a bad-dimension bypass.
"""

import struct

import pytest

from hfl.api.image_validator import (
    _parse_dimensions,
    _parse_gif_dimensions,
    _parse_webp_dimensions,
    decode_base64_image,
    validate_image,
)
from hfl.exceptions import ValidationError as APIValidationError

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _webp(chunk: bytes, body: bytes) -> bytes:
    """RIFF/WEBP container with a given 4-byte chunk type and body."""
    payload = b"WEBP" + chunk + body
    return b"RIFF" + struct.pack("<I", len(payload)) + payload


def _webp_vp8l(width: int, height: int) -> bytes:
    """Minimal VP8L WebP encoding ``width``/``height`` per the 14-bit layout."""
    vw, vh = width - 1, height - 1
    b0 = vw & 0xFF
    b1 = ((vw >> 8) & 0x3F) | ((vh & 0x03) << 6)
    b2 = (vh >> 2) & 0xFF
    b3 = (vh >> 10) & 0x0F
    body = bytes([0]) * 4 + bytes([0x2F, b0, b1, b2, b3]) + bytes(8)
    return _webp(b"VP8L", body)


def _webp_vp8x(width: int, height: int) -> bytes:
    # Parser reads width-1 as 24-bit LE at data offsets 24-26 and height-1 at
    # 27-29. The body begins at data offset 16, so 8 bytes of lead-in put the
    # width field exactly at offset 24.
    vw, vh = width - 1, height - 1
    body = (
        bytes(8)
        + bytes([vw & 0xFF, (vw >> 8) & 0xFF, (vw >> 16) & 0xFF])
        + bytes([vh & 0xFF, (vh >> 8) & 0xFF, (vh >> 16) & 0xFF])
    )
    return _webp(b"VP8X", body)


class TestTruncatedHeaders:
    def test_png_truncated_before_ihdr(self):
        # Passes the size gate (>=10) and sniffs as PNG, but is too short
        # for the IHDR dimension fields.
        data = PNG_SIG + b"\x00\x00\x00\x00"  # 12 bytes < 24
        with pytest.raises(APIValidationError, match="truncated before IHDR"):
            validate_image(data)

    def test_webp_truncated(self):
        data = b"RIFF\x00\x00\x00\x00WEBP"  # 12 bytes, sniffs webp, < 30
        with pytest.raises(APIValidationError, match="WebP file truncated"):
            validate_image(data)

    def test_gif_parser_truncated_direct(self):
        # validate_image's size gate (>=10) and the GIF parser guard (>=10)
        # coincide, so the truncated-GIF branch is only reachable directly.
        with pytest.raises(APIValidationError, match="GIF file truncated"):
            _parse_gif_dimensions(b"GIF89a\x01")


class TestWebPSubFormats:
    def test_vp8l_lossless_dimensions(self):
        info = validate_image(_webp_vp8l(64, 48))
        assert info.mime == "image/webp"
        assert (info.width, info.height) == (64, 48)

    def test_vp8x_extended_dimensions(self):
        info = validate_image(_webp_vp8x(120, 90))
        assert info.mime == "image/webp"
        assert (info.width, info.height) == (120, 90)

    def test_invalid_vp8l_signature_rejected(self):
        bad = _webp_vp8l(16, 16)
        bad = bad[:20] + b"\x00" + bad[21:]  # corrupt the 0x2F signature byte
        with pytest.raises(APIValidationError, match="Invalid VP8L signature"):
            validate_image(bad)

    def test_unsupported_webp_subformat_rejected(self):
        data = _webp(b"XXXX", bytes(20))
        with pytest.raises(APIValidationError, match="Unsupported WebP sub-format"):
            validate_image(data)


class TestJpegScanning:
    def test_jpeg_no_sof_segment_rejected(self):
        # SOI + APP0 segment + an embedded SOI marker + filler, but never a
        # SOF marker: exercises segment-skip, SOI/EOI-skip and non-FF walk.
        data = (
            b"\xff\xd8"  # SOI
            + b"\xff\xe0\x00\x04\x00\x00"  # APP0, length=4 (2 payload bytes)
            + b"\xff\xd8"  # stray SOI marker -> skipped
            + b"\x00" * 24  # non-FF filler so the walk runs off the end
        )
        with pytest.raises(APIValidationError, match="no SOF segment"):
            validate_image(data)


class TestMalformedDimensions:
    def test_unknown_mime_returns_zero(self):
        # An unhandled MIME falls through to (0, 0); validate_image turns that
        # into a clean "could not parse dimensions" rejection.
        assert _parse_dimensions(b"whatever", "image/tiff") == (0, 0)

    def test_webp_truncated_surfaces_as_validation_error(self):
        # The length guard raises APIValidationError (not a raw IndexError),
        # so a truncated WebP can never reach the engine as a 500.
        with pytest.raises(APIValidationError):
            _parse_webp_dimensions(b"RIFF\x00\x00\x00\x00WEBPVP8 ")


class TestDataUriEdges:
    def test_data_uri_missing_comma_rejected(self):
        with pytest.raises(APIValidationError, match="missing comma separator"):
            decode_base64_image("data:image/png;base64AAAA")

    def test_data_uri_non_image_mime_rejected(self):
        with pytest.raises(APIValidationError, match="must be image/"):
            decode_base64_image("data:text/html;base64,AAAA")

    def test_data_uri_without_base64_marker_rejected(self):
        with pytest.raises(APIValidationError, match="must declare"):
            decode_base64_image("data:image/png,AAAA")
