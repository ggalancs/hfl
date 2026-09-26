# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``HFL_MCP_AUTOLOAD`` is read at server start. ``autoload_servers`` existed
but nothing called it, so the variable did nothing (local audit D27)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def calls(monkeypatch):
    seen: list[str] = []

    async def autoload() -> list[str]:
        seen.append("autoload")
        return ["fs"]

    class Client:
        async def disconnect_all(self) -> None:
            seen.append("disconnect_all")

    monkeypatch.setattr("hfl.mcp.client.autoload_servers", autoload)
    monkeypatch.setattr("hfl.mcp.client.get_client", lambda: Client())
    return seen


def test_not_set_nothing_is_connected(monkeypatch, calls) -> None:
    from hfl.api.server import app

    monkeypatch.delenv("HFL_MCP_AUTOLOAD", raising=False)
    with TestClient(app):
        pass
    assert calls == []


def test_set_the_servers_connect_at_start_and_disconnect_at_stop(
    monkeypatch, tmp_path, calls
) -> None:
    from hfl.api.server import app

    monkeypatch.setenv("HFL_MCP_AUTOLOAD", str(tmp_path / "mcp.json"))
    with TestClient(app):
        assert calls == ["autoload"]
    assert calls == ["autoload", "disconnect_all"]


def test_a_broken_config_never_stops_the_server(monkeypatch, tmp_path) -> None:
    from hfl.api.server import app

    async def broken() -> list[str]:
        raise KeyError("servers")

    monkeypatch.setattr("hfl.mcp.client.autoload_servers", broken)
    monkeypatch.setenv("HFL_MCP_AUTOLOAD", str(tmp_path / "mcp.json"))
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
