# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ``hfl.tools.web_fetch`` SSRF guard + HTML extraction."""

from __future__ import annotations

import socket

import httpx
import pytest

from hfl.tools import web_fetch as wf

SAMPLE_HTML = """
<!doctype html>
<html>
<head><title> Example Page </title></head>
<body>
<h1>Heading</h1>
<p>A first paragraph of content.</p>
<p>Second <em>paragraph</em> with a link <a href="https://example.com/doc">doc</a>.</p>
<script>window.foo = 1;</script>
<style>.x {}</style>
</body>
</html>
"""


def _public_getaddrinfo(host, port, *args, **kwargs):
    """Pretend every hostname resolves to a public routable address."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]


def _private_getaddrinfo(host, port, *args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]


def _metadata_getaddrinfo(host, port, *args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 0))]


# ----------------------------------------------------------------------
# SSRF guard
# ----------------------------------------------------------------------


class TestSSRFGuard:
    async def test_http_scheme_accepted(self, monkeypatch):
        monkeypatch.setattr(wf.socket, "getaddrinfo", _public_getaddrinfo)
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text=SAMPLE_HTML))
        original = httpx.AsyncClient

        def _factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(wf.httpx, "AsyncClient", _factory)

        doc = await wf.fetch("https://example.com/page")
        assert doc["title"] == "Example Page"

    async def test_ftp_scheme_rejected(self):
        with pytest.raises(wf.WebFetchError):
            await wf.fetch("ftp://example.com/file")

    async def test_file_scheme_rejected(self):
        with pytest.raises(wf.WebFetchError):
            await wf.fetch("file:///etc/passwd")

    async def test_loopback_rejected(self, monkeypatch):
        monkeypatch.setattr(wf.socket, "getaddrinfo", _private_getaddrinfo)
        with pytest.raises(wf.WebFetchError):
            await wf.fetch("https://example.com/")

    async def test_metadata_endpoint_rejected(self, monkeypatch):
        monkeypatch.setattr(wf.socket, "getaddrinfo", _metadata_getaddrinfo)
        with pytest.raises(wf.WebFetchError):
            await wf.fetch("https://metadata.example/")

    async def test_dns_failure_rejected(self, monkeypatch):
        def _bad(host, port, *a, **k):
            raise OSError("no DNS")

        monkeypatch.setattr(wf.socket, "getaddrinfo", _bad)
        with pytest.raises(wf.WebFetchError):
            await wf.fetch("https://example.com/")

    async def test_empty_url_rejected(self):
        with pytest.raises(wf.WebFetchError):
            await wf.fetch("")

    async def test_missing_hostname_rejected(self):
        with pytest.raises(wf.WebFetchError):
            await wf.fetch("https:///path")


# ----------------------------------------------------------------------
# HTML extraction
# ----------------------------------------------------------------------


class TestHTMLExtraction:
    async def test_extracts_title_content_links(self, monkeypatch):
        monkeypatch.setattr(wf.socket, "getaddrinfo", _public_getaddrinfo)
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text=SAMPLE_HTML))
        original = httpx.AsyncClient

        def _factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(wf.httpx, "AsyncClient", _factory)

        doc = await wf.fetch("https://example.com/")
        assert doc["title"] == "Example Page"
        assert "paragraph of content" in doc["content"]
        assert "window.foo" not in doc["content"]  # script body stripped
        assert doc["links"] == ["https://example.com/doc"]

    async def test_strips_script_and_style(self, monkeypatch):
        monkeypatch.setattr(wf.socket, "getaddrinfo", _public_getaddrinfo)
        html = (
            "<html><head><title>T</title></head>"
            "<body>visible<script>hidden_a</script> "
            "<style>hidden_b</style></body></html>"
        )
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text=html))
        original = httpx.AsyncClient

        def _factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(wf.httpx, "AsyncClient", _factory)

        doc = await wf.fetch("https://example.com/")
        assert doc["content"] == "visible"

    async def test_fragment_is_stripped_from_url(self, monkeypatch):
        monkeypatch.setattr(wf.socket, "getaddrinfo", _public_getaddrinfo)
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text=SAMPLE_HTML))
        original = httpx.AsyncClient

        def _factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(wf.httpx, "AsyncClient", _factory)

        doc = await wf.fetch("https://example.com/page#section")
        assert "#" not in doc["url"]


# ----------------------------------------------------------------------
# Error propagation
# ----------------------------------------------------------------------


class TestErrorPropagation:
    async def test_http_timeout_becomes_fetch_error(self, monkeypatch):
        monkeypatch.setattr(wf.socket, "getaddrinfo", _public_getaddrinfo)

        def raise_timeout(req):
            raise httpx.ConnectTimeout("slow")

        transport = httpx.MockTransport(raise_timeout)
        original = httpx.AsyncClient

        def _factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(wf.httpx, "AsyncClient", _factory)

        with pytest.raises(wf.WebFetchError):
            await wf.fetch("https://example.com/")

    async def test_404_still_returns_extracted_envelope(self, monkeypatch):
        # HTTP-level errors don't raise: the caller still wants the
        # body (a custom 404 page may have useful text). Only
        # protocol-level failures map to WebFetchError.
        monkeypatch.setattr(wf.socket, "getaddrinfo", _public_getaddrinfo)
        transport = httpx.MockTransport(
            lambda req: httpx.Response(404, text="<html><body>Not found.</body></html>")
        )
        original = httpx.AsyncClient

        def _factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(wf.httpx, "AsyncClient", _factory)

        doc = await wf.fetch("https://example.com/")
        assert doc["content"] == "Not found."
