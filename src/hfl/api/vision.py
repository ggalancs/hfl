# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Vision / multimodal conversion helpers (Phase 4, P0-6).

The router layer owns decoding and validation — engines should only
ever see bytes that have already passed through this module. Two
shapes need translating into :class:`hfl.engine.base.ChatMessage`:

1. **Ollama-native** ``messages[i].images: list[str]`` — a list of
   base64-encoded images attached to the message.
2. **OpenAI-compatible** ``content: list[ContentPart]`` — a mix of
   ``{"type":"text", "text":"..."}`` and
   ``{"type":"image_url", "image_url":{"url":"data:image/..."}}``
   parts. Text parts are concatenated in order; image parts are
   decoded, validated and appended to the message's images list.

Both paths land on the same ``ChatMessage`` shape so engines see
one unified multimodal representation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hfl.api.image_validator import validate_image
from hfl.exceptions import ValidationError as APIValidationError

if TYPE_CHECKING:
    pass


# ----------------------------------------------------------------------
# Ollama-native path
# ----------------------------------------------------------------------


def decode_ollama_images(images: list[str] | None) -> list[bytes] | None:
    """Decode + validate the ``images`` array of an Ollama message.

    Each entry must be a base64 string (raw or data-URI). All
    entries are validated through :func:`validate_image` — a
    rejection on any one short-circuits and raises.

    Args:
        images: Raw value from the schema (possibly None).

    Returns:
        A list of bytes in the same order, or None when ``images``
        was None/empty. An empty list is treated the same as None
        to keep the engine contract simple.
    """
    if not images:
        return None
    decoded: list[bytes] = []
    for index, raw in enumerate(images):
        try:
            info = validate_image(raw)
        except APIValidationError as exc:
            raise APIValidationError(f"images[{index}]: {exc}")
        decoded.append(info.data)
    return decoded


# ----------------------------------------------------------------------
# OpenAI content-part path
# ----------------------------------------------------------------------


def split_openai_content(
    content: str | list[Any],
) -> tuple[str, list[bytes] | None]:
    """Split an OpenAI ``content`` field into (text, images).

    Accepts both the legacy string shape and the list-of-parts shape
    introduced by GPT-4 Vision. Text parts are concatenated with a
    single space between them (mirroring OpenAI's own behaviour);
    image parts are validated and decoded.

    Args:
        content: Raw ``ChatCompletionMessage.content`` value — either
            a ``str`` or a ``list`` of Pydantic ``ContentPart``
            instances.

    Returns:
        Tuple ``(text, images)``. ``images`` is ``None`` when the
        content has no image parts, matching the ``ChatMessage``
        contract.

    Raises:
        APIValidationError: An image part has an unsupported URL
            scheme (http://, https://, file:// are rejected — HFL
            is local-first, remote fetching is a separate feature
            not wired yet), or its base64 payload fails validation.
    """
    # Fast path: legacy string shape
    if isinstance(content, str):
        return content, None

    text_pieces: list[str] = []
    images: list[bytes] = []

    for index, part in enumerate(content):
        # Part is a Pydantic model instance — we discriminate by
        # ``type`` rather than isinstance to keep the import graph
        # clean (the schemas module imports helpers; helpers don't
        # import schemas).
        part_type = getattr(part, "type", None)

        if part_type == "text":
            text_pieces.append(getattr(part, "text", "") or "")
            continue

        if part_type == "image_url":
            image_url = getattr(part, "image_url", None)
            if image_url is None:
                raise APIValidationError(
                    f"content[{index}].image_url is required when type=image_url"
                )
            url = getattr(image_url, "url", None) or ""
            # Reject remote URLs — HFL does not fetch arbitrary web
            # content from inside a prompt (would be an SSRF
            # primitive). Only the data-URI form is accepted.
            if not url.startswith("data:"):
                raise APIValidationError(
                    f"content[{index}].image_url.url must be a base64 "
                    "data URI (``data:image/...;base64,...``); http/https "
                    "URLs are not fetched by HFL"
                )
            try:
                info = validate_image(url)
            except APIValidationError as exc:
                raise APIValidationError(f"content[{index}]: {exc}")
            images.append(info.data)
            continue

        # Unknown part type. OpenAI occasionally ships new variants
        # (``input_audio`` etc.); we reject instead of silently
        # ignoring so the client knows to update.
        raise APIValidationError(f"content[{index}].type={part_type!r} is not supported")

    text = " ".join(p for p in text_pieces if p)
    return text, (images or None)
