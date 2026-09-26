# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A chat page in the browser: ``GET /ui`` (and ``/`` for a browser).

One self-contained HTML file (``hfl/ui/chat.html``): no build step, no
external fonts, scripts or CDNs, so it works offline and asks nothing of
anyone but this server. It talks to ``/api/tags`` and ``/api/chat`` like
any other client, so it has no powers of its own; the owner-only routes
refuse it like any other web page.

The page ships with a strict Content-Security-Policy: only its own script
and style run (a nonce per response), and it can only connect back to
this server. Every model reply is escaped before the little markdown the
page renders — code blocks, inline code, bold — is turned into tags.
"""

from __future__ import annotations

import html
import json
import secrets
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_PAGE = Path(__file__).resolve().parents[1] / "ui" / "chat.html"
_KEYS = (
    "model",
    "new_chat",
    "empty",
    "placeholder",
    "send",
    "stop",
    "thinking",
    "no_models",
    "unreachable",
    "failed",
    "api_key",
    *(
        "conversations delete settings system_prompt system_placeholder temperature "
        "model_default attach remove image_error storage_full menu untitled copy copied export"
    ).split(),
)


@lru_cache(maxsize=1)
def _template() -> str:
    return _PAGE.read_text(encoding="utf-8")


def render_chat_page() -> HTMLResponse:
    """The chat page, in the server's language, with a fresh CSP nonce."""
    from hfl.i18n import get_language, t

    text = {key: t(f"ui.{key}") for key in _KEYS}
    nonce = secrets.token_urlsafe(16)
    page = _template().replace("__LANG__", get_language()).replace("__NONCE__", nonce)
    for key, value in text.items():
        page = page.replace(f"__T_{key}__", html.escape(value, quote=True))
    # ``<`` encoded so no string can close the <script> element early.
    page = page.replace("__TEXT_JSON__", json.dumps(text).replace("<", "\\u003c"))
    policy = (
        "default-src 'none'; "
        f"script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
        "connect-src 'self'; img-src 'self' data:; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )
    return HTMLResponse(
        page,
        headers={"Content-Security-Policy": policy, "Cache-Control": "no-store"},
    )


@router.get("/ui", include_in_schema=False)
async def chat_page() -> HTMLResponse:
    return render_chat_page()
