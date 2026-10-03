# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""web_fetch SSRF allow-list, deadline and decompression bound; web_search
log redaction. No network: name resolution and HTTP are faked."""

from __future__ import annotations

import asyncio
import gzip
import logging
import socket
import threading

import httpx
import pytest

from hfl.tools import web_fetch as wf
from hfl.tools import web_search as ws


def _resolves_to(addr: str):
    family = socket.AF_INET6 if ":" in addr else socket.AF_INET

    def _gai(host, port, *args, **kwargs):
        return [(family, socket.SOCK_STREAM, 6, "", (addr, 0))]

    return _gai


def _mock_http(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(wf.httpx, "AsyncClient", _factory)


# ----------------------------------------------------------------------
# Finding 2: only globally routable addresses
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "addr",
    [
        "100.100.100.200",  # Alibaba Cloud metadata (CGNAT range)
        "100.64.0.1",  # CGNAT / Tailscale
        "::ffff:100.100.100.200",  # IPv4-mapped
        "::ffff:127.0.0.1",
        "64:ff9b::a9fe:a9fe",  # NAT64 of 169.254.169.254
        "64:ff9b::6464:64c8",  # NAT64 of 100.100.100.200
        "2002:7f00:1::",  # 6to4 of 127.0.0.1
        "fd00::1",
        "192.0.0.1",
        "198.18.0.1",  # benchmarking
    ],
)
def test_non_global_addresses_are_refused(monkeypatch, addr):
    monkeypatch.setattr(wf.socket, "getaddrinfo", _resolves_to(addr))
    with pytest.raises(wf.WebFetchError):
        wf._resolve_and_validate("http://example.com/")


@pytest.mark.parametrize("addr", ["93.184.216.34", "2606:4700:4700::1111", "64:ff9b::5db8:d822"])
def test_global_addresses_pass(monkeypatch, addr):
    monkeypatch.setattr(wf.socket, "getaddrinfo", _resolves_to(addr))
    _, pinned = wf._resolve_and_validate("http://example.com/")
    assert pinned == addr


async def test_name_resolution_runs_off_the_event_loop(monkeypatch):
    threads: list[threading.Thread] = []

    def _gai(host, port, *args, **kwargs):
        threads.append(threading.current_thread())
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(wf.socket, "getaddrinfo", _gai)
    _mock_http(monkeypatch, lambda req: httpx.Response(200, text="<title>T</title>"))
    await wf.fetch("https://example.com/")
    assert threads and all(t is not threading.main_thread() for t in threads)


# ----------------------------------------------------------------------
# Finding 10: overall deadline, no decompression bomb
# ----------------------------------------------------------------------


async def test_a_trickling_server_hits_the_overall_deadline(monkeypatch):
    monkeypatch.setattr(wf.socket, "getaddrinfo", _resolves_to("93.184.216.34"))
    monkeypatch.setattr(wf, "_OVERALL_DEADLINE_S", 0.5)

    async def _trickle():
        while True:  # a byte well inside every per-read timeout, forever
            await asyncio.sleep(0.05)
            yield b"a"

    _mock_http(monkeypatch, lambda req: httpx.Response(200, content=_trickle()))
    with pytest.raises(wf.WebFetchError, match="timed out"):
        # The harness's own bound: without the fix this fails, not hangs.
        await asyncio.wait_for(wf.fetch("https://example.com/"), timeout=10)


async def test_compressed_bodies_are_not_inflated(monkeypatch):
    monkeypatch.setattr(wf.socket, "getaddrinfo", _resolves_to("93.184.216.34"))
    bomb = gzip.compress(b"\0" * (64 * 1024 * 1024))  # ~64 KiB → 64 MiB
    seen: dict[str, str] = {}

    def handler(req):
        seen["accept-encoding"] = req.headers.get("accept-encoding", "")
        return httpx.Response(200, content=bomb, headers={"content-encoding": "gzip"})

    _mock_http(monkeypatch, handler)
    with pytest.raises(wf.WebFetchError):
        await wf.fetch("https://example.com/", max_bytes=1024 * 1024)
    assert seen["accept-encoding"] == "identity"


# ----------------------------------------------------------------------
# Finding 11: no API key in the log
# ----------------------------------------------------------------------


async def test_serpapi_failure_does_not_log_the_key(monkeypatch, caplog):
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-secret-123")
    original = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(lambda req: httpx.Response(500))
        return original(*args, **kwargs)

    monkeypatch.setattr(ws.httpx, "AsyncClient", _factory)
    with caplog.at_level(logging.DEBUG, logger=ws.logger.name):
        with pytest.raises(ws.WebSearchError):
            await ws.SerpAPIBackend().search("q", 3)
    assert caplog.records, "the failure must still be logged"
    assert "serp-secret-123" not in caplog.text
    assert "500" in caplog.text
