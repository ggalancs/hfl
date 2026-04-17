# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""URL content extraction with SSRF guard (Phase 9 P0).

Ollama's ``/api/web_fetch`` companion to ``web_search``: given a URL
returns ``{"title", "content", "links"}`` extracted from the HTML.

Security posture:

- Only http/https schemes accepted.
- Hostname is resolved *before* the request and rejected if the
  address lands on a private / loopback / link-local range. This
  prevents SSRF against cloud-metadata endpoints (169.254.169.254)
  and internal services.
- Response body is capped at 5 MiB by default.
- Charset is sniffed from the Content-Type header and falls back to
  UTF-8 with replacement errors.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

logger = logging.getLogger(__name__)

__all__ = ["WebFetchError", "fetch"]

_DEFAULT_MAX_BYTES = 5 * 1024 * 1024
_ALLOWED_SCHEMES = frozenset({"http", "https"})


class WebFetchError(ValueError):
    """Raised when a URL is unreachable or rejected by the guard."""


# ----------------------------------------------------------------------
# SSRF guard
# ----------------------------------------------------------------------


def _is_private_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # Unparseable == unsafe.
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _validate_url(url: str) -> str:
    """Reject hostile URLs before httpx even touches the socket."""
    if not isinstance(url, str) or not url:
        raise WebFetchError("URL must be a non-empty string")

    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise WebFetchError(f"URL scheme must be http or https, got {parsed.scheme!r}")
    if not parsed.hostname:
        raise WebFetchError("URL is missing a hostname")

    # Resolve and reject private / loopback / metadata targets.
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except (socket.gaierror, OSError) as exc:
        raise WebFetchError("hostname does not resolve") from exc
    for info in infos:
        addr = info[4][0]
        if _is_private_ip(addr):
            raise WebFetchError("URL resolves to a private or reserved address — refusing to fetch")

    # Normalise: strip fragments (we never need them).
    cleaned = parsed._replace(fragment="")
    return urlunparse(cleaned)


# ----------------------------------------------------------------------
# HTML extraction (no BeautifulSoup — single std-lib parser)
# ----------------------------------------------------------------------


class _DocumentExtractor(HTMLParser):
    """Walk the DOM once and collect title, visible text, links."""

    _SKIP_TAGS = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower == "title":
            self._in_title = True
        elif lower in self._SKIP_TAGS:
            self._skip_depth += 1
        elif lower == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self.links.append(v)

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower == "title":
            self._in_title = False
        elif lower in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        else:
            stripped = data.strip()
            if stripped:
                self.text_parts.append(stripped)


def _extract(body: str) -> dict[str, Any]:
    """Run the HTML parser and collapse whitespace."""
    parser = _DocumentExtractor()
    try:
        parser.feed(body)
    except Exception:  # pragma: no cover — HTMLParser rarely raises
        logger.warning("HTML parsing failed; returning raw truncated text", exc_info=True)
    title = re.sub(r"\s+", " ", "".join(parser.title_parts)).strip()
    content = re.sub(r"\s+", " ", " ".join(parser.text_parts)).strip()
    return {"title": title, "content": content, "links": parser.links}


# ----------------------------------------------------------------------
# Public fetch entrypoint
# ----------------------------------------------------------------------


async def fetch(
    url: str,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Fetch ``url`` and return ``{"title", "content", "links"}``.

    Never raises for HTTP-level errors — a 404 still returns a
    valid envelope with whatever body came back. Raises
    ``WebFetchError`` only for protocol / security / timeout
    failures, which the route turns into a 400.
    """
    cleaned_url = _validate_url(url)
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            max_redirects=3,
        ) as client:
            resp = await client.get(
                cleaned_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (HFL-bot) Python/httpx",
                    "Accept": "text/html,application/xhtml+xml",
                },
            )
            body_bytes = resp.content[:max_bytes]
            charset = resp.charset_encoding or "utf-8"
            try:
                body = body_bytes.decode(charset, errors="replace")
            except LookupError:
                body = body_bytes.decode("utf-8", errors="replace")
    except httpx.HTTPError as exc:
        logger.info("web_fetch HTTP error for %s: %s", cleaned_url, exc)
        raise WebFetchError("URL could not be fetched") from exc

    extracted = _extract(body)
    return {
        "title": extracted["title"],
        "content": extracted["content"],
        "links": extracted["links"],
        "url": cleaned_url,
    }
