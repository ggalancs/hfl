# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A download that fails in transport is retried with huggingface_hub 1.x
(httpx errors) and 2.x (httpx2 errors, separate classes): catching httpx's
alone stopped every retry under 2.x."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", ["httpx", "httpx2"])
def test_a_transport_failure_is_retried(module, monkeypatch, tmp_path) -> None:
    http = pytest.importorskip(module)
    from hfl.hub import downloader

    monkeypatch.setattr("hfl.utils.retry.time.sleep", lambda seconds: None)
    calls: list[int] = []

    def flaky(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise http.ReadError("connection cut mid-download")
        return str(tmp_path / "config.json")

    monkeypatch.setattr(downloader, "hf_hub_download", flaky)
    got = downloader._download_file("o/r", "config.json", None, tmp_path, None)
    assert got == Path(tmp_path / "config.json") and len(calls) == 2


@pytest.mark.parametrize("module", ["httpx", "httpx2"])
def test_an_http_status_error_is_not_retried(module, monkeypatch, tmp_path) -> None:
    """A 404 is an answer: retrying it only delays the real message."""
    http = pytest.importorskip(module)
    from hfl.hub import downloader

    monkeypatch.setattr("hfl.utils.retry.time.sleep", lambda seconds: None)
    calls: list[int] = []

    def missing(**kwargs):
        calls.append(1)
        request = http.Request("GET", "https://huggingface.co/o/r")
        raise http.HTTPStatusError(
            "404", request=request, response=http.Response(404, request=request)
        )

    monkeypatch.setattr(downloader, "hf_hub_download", missing)
    with pytest.raises(http.HTTPStatusError):
        downloader._download_file("o/r", "config.json", None, tmp_path, None)
    assert len(calls) == 1


def test_both_generations_are_retryable_when_both_are_installed() -> None:
    from hfl.hub import downloader

    for module in ("httpx", "httpx2"):
        try:
            error = importlib.import_module(module).TransportError
        except ImportError:
            continue
        assert error in downloader._RETRYABLE_EXCEPTIONS, module
