# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl websearch``: the web search provider chosen once, saved in HFL's
home, read by every server (the desktop app's included)."""

from __future__ import annotations

import asyncio
import json
import os
import stat

import pytest
from typer.testing import CliRunner

from hfl.cli.commands import websearch as setup
from hfl.tools import web_search as ws

VARS = ("HFL_WEB_SEARCH_BACKEND", "HFL_SEARXNG_URL", *setup.KEYED.values())


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """An empty HFL home and no search settings in the environment."""
    from hfl.config import config

    monkeypatch.setattr(config, "home_dir", tmp_path)
    for var in VARS:
        monkeypatch.delenv(var, raising=False)
    return tmp_path


class TestSavedSettings:
    def test_saved_readable_by_the_owner_only_and_merged(self, home):
        path = ws.save_settings({"HFL_WEB_SEARCH_BACKEND": "brave", "BRAVE_API_KEY": "k1"})
        assert path == home / "web_search.json"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        ws.save_settings({"BRAVE_API_KEY": None, "HFL_SEARXNG_URL": "http://127.0.0.1:8888"})
        assert ws.saved_settings() == {
            "HFL_WEB_SEARCH_BACKEND": "brave",
            "HFL_SEARXNG_URL": "http://127.0.0.1:8888",
        }

    def test_unreadable_or_absent_is_nothing(self, home):
        assert ws.saved_settings() == {}
        (home / "web_search.json").write_text("not json")
        assert ws.saved_settings() == {}
        (home / "web_search.json").write_text("[1, 2]")
        assert ws.saved_settings() == {}

    def test_the_environment_wins_over_the_file(self, monkeypatch):
        ws.save_settings({"BRAVE_API_KEY": "from-file"})
        assert ws._setting("BRAVE_API_KEY") == "from-file"
        monkeypatch.setenv("BRAVE_API_KEY", "from-env")
        assert ws._setting("BRAVE_API_KEY") == "from-env"

    def test_the_saved_backend_is_the_chain(self):
        ws.save_settings({"HFL_WEB_SEARCH_BACKEND": "searxng", "HFL_SEARXNG_URL": "http://x"})
        assert [b.name for b in ws.backend_chain()] == ["searxng"]

    def test_a_saved_key_joins_the_chain_before_duckduckgo(self):
        ws.save_settings({"TAVILY_API_KEY": "t"})
        assert [b.name for b in ws.backend_chain()] == ["tavily", "duckduckgo"]

    def test_a_backend_reads_its_saved_key(self, monkeypatch):
        """The server started by the desktop app has no shell environment:
        the key comes from the file."""
        import httpx

        ws.save_settings({"BRAVE_API_KEY": "saved-key"})
        seen: dict[str, str] = {}

        class _Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, params=None, headers=None, **k):
                seen.update(headers or {})
                return httpx.Response(
                    200, json={"web": {"results": []}}, request=httpx.Request("GET", url)
                )

        monkeypatch.setattr(ws.httpx, "AsyncClient", _Client)
        asyncio.run(ws.BraveBackend().search("q", 1))
        assert "saved-key" in json.dumps(seen)


class TestSearxngIsTheUsers:
    """HFL connects to a SearXNG the user runs; it never installs or starts
    one (the owner's rule, 2026-10-25)."""

    def test_a_port_is_this_machines(self):
        assert setup.searxng_url(8899, None) == "http://127.0.0.1:8899"

    def test_a_url_wins_over_a_port(self):
        assert setup.searxng_url(8899, "http://10.0.0.2:8080/") == "http://10.0.0.2:8080"

    def test_neither_says_how(self):
        with pytest.raises(setup.SetupError, match="does not install it"):
            setup.searxng_url(None, None)

    def test_not_a_port(self):
        with pytest.raises(setup.SetupError, match="not a port"):
            setup.searxng_url(70000, None)

    def test_one_that_does_not_answer_is_not_saved(self, monkeypatch):
        import httpx

        def refuse(url, timeout=10):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(setup, "probe_searxng", refuse)
        with pytest.raises(setup.SetupError, match="no SearXNG answered"):
            setup.use_searxng("http://127.0.0.1:8899")
        assert ws.saved_settings() == {}

    def test_setting_it_up_starts_no_process(self, monkeypatch):
        """No docker pull, no docker run, nothing: a process started here
        would be HFL installing or running SearXNG."""
        import subprocess as sp

        from hfl.cli import main

        def forbidden(*a, **k):
            raise AssertionError(f"a process was started: {a}")

        for name in ("run", "Popen", "call", "check_call", "check_output"):
            monkeypatch.setattr(sp, name, forbidden)
        monkeypatch.setattr(setup, "probe_searxng", lambda url, timeout=10: 3)
        for args in (["--port", "8899"], ["--url", "http://10.0.0.2:8080"]):
            result = CliRunner().invoke(main.app, ["websearch", "searxng", *args])
            assert result.exit_code == 0, result.output
        result = CliRunner().invoke(main.app, ["websearch", "searxng"])
        assert result.exit_code == 1 and "does not install it" in " ".join(result.output.split())


class TestProbe:
    def test_json_disabled_is_explained(self, monkeypatch):
        import httpx

        monkeypatch.setattr(
            setup.httpx, "get",
            lambda *a, **k: httpx.Response(403, request=httpx.Request("GET", "http://x")),
        )  # fmt: skip
        with pytest.raises(setup.SetupError, match="formats"):
            setup.probe_searxng("http://x")

    def test_counts_the_results(self, monkeypatch):
        import httpx

        monkeypatch.setattr(
            setup.httpx, "get",
            lambda *a, **k: httpx.Response(
                200, json={"results": [{}, {}]}, request=httpx.Request("GET", "http://x")
            ),
        )  # fmt: skip
        assert setup.probe_searxng("http://x") == 2


class TestKeys:
    def test_a_key_is_tried_then_saved(self, monkeypatch):
        tried: list[str | None] = []

        async def search(self, query, max_results):
            tried.append(os.environ.get("TAVILY_API_KEY"))
            return [{"title": "t", "url": "u", "content": "c"}]

        monkeypatch.setattr(ws.TavilyBackend, "search", search)
        assert setup.use_key("Tavily", "  tk-123  ") == 1
        assert tried == ["tk-123"]
        assert os.environ.get("TAVILY_API_KEY") is None  # not left in the environment
        assert ws.saved_settings() == {
            "HFL_WEB_SEARCH_BACKEND": "tavily",
            "TAVILY_API_KEY": "tk-123",
        }

    def test_a_refused_key_is_not_saved(self, monkeypatch):
        async def search(self, query, max_results):
            raise ws.WebSearchUpstreamError("401 unauthorized")

        monkeypatch.setattr(ws.BraveBackend, "search", search)
        with pytest.raises(setup.SetupError, match="refused"):
            setup.use_key("brave", "bad")
        assert ws.saved_settings() == {}

    def test_unchecked(self):
        assert setup.use_key("exa", "ek", check=False) is None
        assert ws.saved_settings()["EXA_API_KEY"] == "ek"

    @pytest.mark.parametrize(
        ("provider", "key", "match"), [("google", "k", "unknown"), ("brave", " ", "empty")]
    )
    def test_refused_input(self, provider, key, match):
        with pytest.raises(setup.SetupError, match=match):
            setup.use_key(provider, key, check=False)


class TestStatusAndOff:
    def test_status_masks_keys_and_names_the_source(self, monkeypatch):
        ws.save_settings({"HFL_WEB_SEARCH_BACKEND": "brave", "BRAVE_API_KEY": "abcdefghijkl1234"})

        async def search(query, max_results=5):
            return {"results": [{}], "backend": "brave"}

        monkeypatch.setattr(ws, "search", search)
        report = setup.status()
        rows = {row[0]: row for row in report["settings"]}
        assert rows["BRAVE_API_KEY"][1] == "…1234"
        assert rows["BRAVE_API_KEY"][2].endswith("web_search.json")
        assert report["chain"] == ["brave"] and report["ok"] is True

    def test_off_forgets_the_file(self, home):
        ws.save_settings({"HFL_WEB_SEARCH_BACKEND": "brave"})
        assert setup.turn_off() is True
        assert not (home / "web_search.json").exists()
        assert setup.turn_off() is False


class TestCommands:
    def test_key_from_stdin_never_from_argv(self, monkeypatch):
        from hfl.cli import main

        result = CliRunner().invoke(
            main.app, ["websearch", "key", "serpapi", "--no-check"], input="sk-from-stdin\n"
        )
        assert result.exit_code == 0, result.output
        assert ws.saved_settings()["SERPAPI_API_KEY"] == "sk-from-stdin"
        assert "sk-from-stdin" not in result.output

    def test_searxng_with_a_url(self, monkeypatch):
        from hfl.cli import main

        monkeypatch.setattr(setup, "probe_searxng", lambda url, timeout=10: 4)
        result = CliRunner().invoke(
            main.app, ["websearch", "searxng", "--url", "http://10.0.0.2:8080/"]
        )
        assert result.exit_code == 0, result.output
        assert ws.saved_settings()["HFL_SEARXNG_URL"] == "http://10.0.0.2:8080"

    def test_a_setup_error_is_one_line_and_exit_1(self, monkeypatch):
        from hfl.cli import main

        monkeypatch.setattr(
            setup,
            "probe_searxng",
            lambda url, timeout=10: (_ for _ in ()).throw(
                setup.SetupError("refused the JSON format")
            ),
        )
        result = CliRunner().invoke(main.app, ["websearch", "searxng", "--port", "8899"])
        assert result.exit_code == 1
        assert "JSON format" in result.output and "Traceback" not in result.output

    def test_status_fails_when_the_test_search_does(self, monkeypatch):
        from hfl.cli import main

        async def search(query, max_results=5):
            raise ws.WebSearchUpstreamError("DuckDuckGo refused the search")

        monkeypatch.setattr(ws, "search", search)
        result = CliRunner().invoke(main.app, ["websearch", "status"])
        assert result.exit_code == 1
        assert "DuckDuckGo refused" in result.output
