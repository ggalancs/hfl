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
- The validated IP is *pinned* for the actual connection (the request
  dials the vetted address with the original Host header + TLS SNI), so
  httpx cannot re-resolve at connect time — closing the DNS-rebinding
  TOCTOU window.
- Redirects are followed manually, re-validated, and re-pinned on every
  hop, so a public URL cannot 302 the fetch onto an internal/metadata
  address.
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
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

logger = logging.getLogger(__name__)

__all__ = ["WebFetchError", "fetch"]

_DEFAULT_MAX_BYTES = 5 * 1024 * 1024
_ALLOWED_SCHEMES = frozenset({"http", "https"})
_MAX_REDIRECTS = 3


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


def _resolve_and_validate(url: str) -> tuple[str, str]:
    """Validate + resolve a URL, returning ``(cleaned_url, pinned_ip)``.

    Rejects hostile URLs before httpx touches the socket *and* returns a
    concrete validated IP to connect to. Reusing that IP for the actual
    request closes the DNS-rebinding TOCTOU window: without it, httpx
    re-resolves the hostname at connect time, and an attacker controlling
    authoritative DNS (TTL 0) could answer with a public address here and
    a private/metadata address (169.254.169.254 / 127.0.0.1) to httpx,
    bypassing the guard.
    """
    if not isinstance(url, str) or not url:
        raise WebFetchError("URL must be a non-empty string")

    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise WebFetchError(f"URL scheme must be http or https, got {parsed.scheme!r}")
    if not parsed.hostname:
        raise WebFetchError("URL is missing a hostname")

    # Resolve and reject private / loopback / metadata targets. Every
    # returned address must be public; we pin the first one.
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except (socket.gaierror, OSError) as exc:
        raise WebFetchError("hostname does not resolve") from exc
    pinned_ip: str | None = None
    for info in infos:
        addr = str(info[4][0])
        if _is_private_ip(addr):
            raise WebFetchError("URL resolves to a private or reserved address — refusing to fetch")
        if pinned_ip is None:
            pinned_ip = addr
    if pinned_ip is None:
        raise WebFetchError("hostname does not resolve")

    # Normalise: strip fragments (we never need them).
    cleaned = parsed._replace(fragment="")
    return urlunparse(cleaned), pinned_ip


def _pinned_request(cleaned_url: str, pinned_ip: str) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build the connection target that forces httpx to dial ``pinned_ip``.

    Returns ``(connect_url, host_headers, extensions)``. The URL host is
    swapped for the validated IP literal while the original authority is
    preserved in the ``Host`` header and (for https) the TLS SNI /
    certificate hostname, so virtual hosting and certificate validation
    keep working against the real hostname instead of the bare IP.
    """
    parsed = urlparse(cleaned_url)
    host = parsed.hostname or ""
    hh_host = f"[{host}]" if ":" in host else host
    host_header = hh_host if parsed.port is None else f"{hh_host}:{parsed.port}"

    ip_lit = f"[{pinned_ip}]" if ":" in pinned_ip else pinned_ip
    netloc = ip_lit if parsed.port is None else f"{ip_lit}:{parsed.port}"
    # Preserve any userinfo (HTTP Basic credentials) verbatim so httpx still
    # derives the Authorization header — the IP-only rewrite would otherwise
    # silently drop ``user:pass@`` and break auth-protected fetches.
    if "@" in parsed.netloc:
        netloc = parsed.netloc.rsplit("@", 1)[0] + "@" + netloc
    connect_url = urlunparse(parsed._replace(netloc=netloc))

    extensions: dict[str, Any] = {}
    if parsed.scheme.lower() == "https":
        # Verify the certificate against the real hostname, not the IP.
        extensions["sni_hostname"] = host
    return connect_url, {"Host": host_header}, extensions


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
    cleaned_url, pinned_ip = _resolve_and_validate(url)
    headers = {
        "User-Agent": "Mozilla/5.0 (HFL-bot) Python/httpx",
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        # Redirects are followed manually so every hop is re-validated by the
        # SSRF guard *and* pinned to its validated IP. httpx's own
        # follow_redirects (and its connect-time DNS re-resolution) would
        # otherwise chase / rebind onto 169.254.169.254 / 127.0.0.1.
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            current, current_ip = cleaned_url, pinned_ip
            body_bytes = b""
            charset = "utf-8"
            for _ in range(_MAX_REDIRECTS + 1):
                connect_url, host_header, extensions = _pinned_request(current, current_ip)
                # Stream so the body is read incrementally with a hard byte
                # budget: a non-streaming ``get`` buffers the ENTIRE response
                # into memory before slicing, so ``max_bytes`` capped only the
                # parser input, not RAM — a multi-GB body (the model controls
                # the URL) would OOM the process regardless. (SEC)
                async with client.stream(
                    "GET",
                    connect_url,
                    headers={**headers, **host_header},
                    extensions=extensions,
                ) as resp:
                    location = resp.headers.get("location") if resp.is_redirect else None
                    if location:
                        # Redirect: only the Location matters — never read the
                        # (possibly huge) redirect body. Re-run the full scheme +
                        # private-IP guard and re-pin against the logical URL
                        # before following.
                        current, current_ip = _resolve_and_validate(urljoin(current, location))
                        continue
                    # Terminal hop (non-redirect, or redirect without Location):
                    # pull the body until the budget is reached, then stop.
                    charset = resp.charset_encoding or "utf-8"
                    total = 0
                    chunks: list[bytes] = []
                    async for chunk in resp.aiter_bytes():
                        chunks.append(chunk)
                        total += len(chunk)
                        if total >= max_bytes:
                            break
                    body_bytes = b"".join(chunks)[:max_bytes]
                    break
            else:
                raise WebFetchError("too many redirects")
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
