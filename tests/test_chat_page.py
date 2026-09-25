# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The chat page: served to browsers, never in the way of clients."""

from __future__ import annotations

import re
import shutil
import subprocess
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app

BROWSER = {"Accept": "text/html,application/xhtml+xml,*/*;q=0.8"}


class _Page(HTMLParser):
    """The page's script text, and everything else (text and attributes)."""

    def __init__(self) -> None:
        super().__init__()
        self._in_script = False
        self.script = ""
        self.outside_script = ""

    @classmethod
    def of(cls, html: str) -> "_Page":
        page = cls()
        page.feed(html)
        return page

    def handle_starttag(self, tag, attrs):
        self._in_script = tag == "script"
        self.outside_script += " ".join(f"{k}={v}" for k, v in attrs if v) + " "

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False

    def handle_data(self, data):
        if self._in_script:
            self.script += data
        else:
            self.outside_script += data


@pytest.fixture
def client(temp_config):
    return TestClient(app)


def test_clients_keep_their_json_at_the_root(client):
    assert client.get("/").json() == {"status": "hfl is running"}
    assert client.get("/", headers={"Accept": "application/json"}).json() == {
        "status": "hfl is running"
    }


@pytest.mark.parametrize(("path", "headers"), [("/", BROWSER), ("/ui", {})])
def test_a_browser_gets_the_page(client, path, headers):
    response = client.get(path, headers=headers)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>HFL</title>" in response.text
    page = _Page.of(response.text)
    assert "__" not in page.outside_script  # every placeholder filled


def test_the_policy_allows_only_its_own_script_and_this_server(client):
    response = client.get("/ui")
    policy = response.headers["content-security-policy"]
    nonce = re.search(r"'nonce-([^']+)'", policy).group(1)
    assert f'<script nonce="{nonce}">' in response.text
    assert f'<style nonce="{nonce}">' in response.text
    assert "default-src 'none'" in policy and "connect-src 'self'" in policy
    assert "frame-ancestors 'none'" in policy and "unsafe-inline" not in policy
    assert client.get("/ui").headers["content-security-policy"] != policy  # fresh nonce


def test_nothing_is_loaded_from_elsewhere(client):
    page = client.get("/ui").text
    assert not re.search(r"""(src|href)=["']?(https?:)?//""", page)
    assert "@import" not in page and "url(http" not in page


def test_it_speaks_the_servers_language(client, monkeypatch):
    from hfl.i18n import get_language

    monkeypatch.setenv("HFL_LANG", "es")
    get_language.cache_clear()
    try:
        page = client.get("/ui").text
    finally:
        get_language.cache_clear()  # not Spanish for the tests after this one
    assert '<html lang="es">' in page and "Nueva conversación" in page


def test_an_api_key_does_not_lock_the_page_itself(client):
    from hfl.api.state import get_state

    state = get_state()
    state.api_key = "s3cret"
    try:
        assert client.get("/ui").status_code == 200
        assert client.get("/api/tags").status_code == 401  # the API still asks
    finally:
        state.api_key = None


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_model_output_is_escaped_before_rendering(client):
    """The page's own render(): markup a model writes must come out as text."""
    page = client.get("/ui").text
    script = _Page.of(page).script
    functions = re.search(r"(function escapeHtml[\s\S]*?)\nfunction bubble", script).group(1)
    js = (
        'const T = {thinking: "Thinking"};\n' + functions + "\n"
        "const out = render('<img src=x onerror=alert(1)> **bold** `a<b`\\n\\n"
        "```html\\n<script>x</script>\\n```');\n"
        "process.stdout.write(out);"
    )
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30).stdout
    assert "<img" not in out and "<script>" not in out
    assert "&lt;img src=x onerror=alert(1)&gt;" in out
    assert "<strong>bold</strong>" in out and "<code>a&lt;b</code>" in out
    assert "<pre><code>&lt;script&gt;x&lt;/script&gt;\n</code></pre>" in out


@pytest.mark.parametrize("lang", ["en", "es"])
def test_every_placeholder_is_filled(client, monkeypatch, lang):
    """A key missing from ``_KEYS`` or a locale shows up as ``__T_x__``."""
    from hfl.i18n import get_language

    monkeypatch.setenv("HFL_LANG", lang)
    get_language.cache_clear()
    try:
        page = client.get("/ui").text
    finally:
        get_language.cache_clear()
    assert not re.findall(r"__[A-Z]+[A-Za-z_]*__", page)
    assert "ui." not in re.sub(r"<script.*</script>", "", page, flags=re.S)


def test_no_inline_event_handlers(client):
    """The policy runs only the nonce'd script: an ``onclick=`` attribute
    would be dead, silently."""
    page = client.get("/ui").text
    assert not re.findall(r"<[^>]*\son[a-z]+\s*=", page)
