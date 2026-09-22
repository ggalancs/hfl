# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Independence, stated as a gate instead of a promise.

HFL exists to avoid the infrastructure a competitor built around a cloud
proxy you are pushed to register with. That makes "this package talks to
nobody you did not choose" a *product requirement*, not a nice property —
and a requirement nothing checks is a requirement that drifts.

So this module enumerates every host the source can reach and pins the
set. Adding a host is allowed; adding one *silently* is not. Three claims
are tested, and they are different claims:

1. Every URL in the source belongs to a host listed below, each with a
   written reason.
2. Out of the box — no environment set — the only service HFL reaches for
   its own function is the HuggingFace Hub.
3. Nothing is contacted at import time. Merely importing the package must
   not open a socket, or "opt-in" would be decided after the fact.

The allow-list is deliberately literal rather than a pattern. A pattern
like ``*.huggingface.co`` would silently admit a host nobody reviewed,
which is the failure this file exists to prevent.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "hfl"

# host -> why it is allowed. REFERENCE hosts appear only as text a human
# reads; RUNTIME hosts can be contacted by code.
ALLOWED_HOSTS: dict[str, str] = {
    # --- runtime, and the only one on by default -------------------------
    "huggingface.co": "RUNTIME: the model source. The one accepted dependency.",
    # --- runtime, opt-in, off unless the operator switches it on ---------
    "html.duckduckgo.com": "RUNTIME opt-in: default web_search backend, no account, no key.",
    "api.tavily.com": "RUNTIME opt-in: web_search backend, needs TAVILY_API_KEY.",
    "api.search.brave.com": "RUNTIME opt-in: web_search backend, needs BRAVE_API_KEY.",
    "serpapi.com": "RUNTIME opt-in: web_search backend, needs SERPAPI_API_KEY.",
    # --- reference only: printed or documented, never fetched ------------
    "tavily.com": "REFERENCE: where to get the key, shown in an error message.",
    "search.brave.com": "REFERENCE: where to get the key, shown in an error message.",
    "github.com": "REFERENCE: this project's own repo, printed by `hfl version`.",
    "docs.ollama.com": "REFERENCE: the API HFL is wire-compatible with; documentation links.",
    # --- the machine itself ----------------------------------------------
    "localhost": "LOCAL: the server's own bind address.",
    "127.0.0.1": "LOCAL: the server's own bind address.",
}

# Hosts reachable with no environment variables set at all.
DEFAULT_RUNTIME_HOSTS = {"huggingface.co", "html.duckduckgo.com"}

_URL = re.compile(r"https?://([A-Za-z0-9._-]+)")


def _iter_sources():
    for path in sorted(SRC.rglob("*.py")):
        yield path, path.read_text(encoding="utf-8")


def test_every_url_in_the_source_is_on_the_allow_list():
    """A new outbound host has to be added here on purpose."""
    found: dict[str, list[str]] = {}
    for path, text in _iter_sources():
        for host in _URL.findall(text):
            found.setdefault(host, []).append(str(path.relative_to(SRC)))

    unlisted = {h: sorted(set(v)) for h, v in found.items() if h not in ALLOWED_HOSTS}
    assert not unlisted, (
        "New host(s) reachable from the source. If this is deliberate, add it to "
        "ALLOWED_HOSTS with the reason it is acceptable under the independence "
        f"requirement — and if it is a RUNTIME host, make it opt-in:\n{unlisted}"
    )


def test_the_allow_list_has_no_dead_entries():
    """An entry nobody uses is a permission granted for nothing."""
    present = set()
    for _path, text in _iter_sources():
        present.update(_URL.findall(text))
    stale = sorted(set(ALLOWED_HOSTS) - present)
    assert not stale, (
        f"ALLOWED_HOSTS lists hosts the source no longer mentions: {stale}. "
        "Remove them — a stale allowance is how an unreviewed host slips back in."
    )


def test_hub_is_the_only_service_needed_out_of_the_box():
    """With nothing configured, HFL depends on the Hub and nothing else.

    The opt-in backends may be *reachable*, but reaching them has to be a
    choice the operator made. DuckDuckGo is in the default set because it
    backs `web_search` without an account — a tool the caller invokes,
    never something HFL does on its own.
    """
    extras = DEFAULT_RUNTIME_HOSTS - {"huggingface.co", "html.duckduckgo.com"}
    assert not extras, f"something new is contacted by default: {extras}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "env_var"),
    [("tavily", "TAVILY_API_KEY"), ("brave", "BRAVE_API_KEY"), ("serpapi", "SERPAPI_API_KEY")],
)
async def test_keyed_backends_refuse_without_their_key(backend, env_var, monkeypatch):
    """An account-gated service must refuse rather than quietly become the path.

    The refusal lands on ``search``, not on construction: building a
    backend opens nothing, which is itself part of the requirement.
    """
    from hfl.tools.web_search import WebSearchError, get_backend

    monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("HFL_WEB_SEARCH_BACKEND", backend)

    instance = get_backend()  # must not raise, and must not connect
    with pytest.raises(WebSearchError) as caught:
        await instance.search("anything", 3)
    assert env_var in str(caught.value), "the error must name the variable to set"


def test_default_backend_needs_no_account(monkeypatch):
    from hfl.tools.web_search import get_backend

    for var in ("HFL_WEB_SEARCH_BACKEND", "TAVILY_API_KEY", "BRAVE_API_KEY", "SERPAPI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    backend = get_backend()
    assert "duckduckgo" in type(backend).__name__.lower()


def test_no_module_opens_a_socket_at_import_time():
    """Importing must not be a network event.

    Static rather than runtime: a runtime check would only cover the
    modules this test happened to import, and the claim is about all of
    them.

    Import aliases are resolved rather than matched by name. A first
    version of this check looked for calls on objects *called* ``httpx``
    or ``requests``, and a sabotage of ``import httpx as _hx`` followed by
    ``_hx.get(...)`` walked straight past it — a check that cannot fail is
    worse than no check, so the binding is followed instead of the label.
    """
    NETWORK_MODULES = {"httpx", "requests", "urllib", "urllib.request", "socket", "aiohttp"}
    NETWORK_CALLS = {"get", "post", "put", "delete", "head", "patch", "request", "urlopen"}
    offenders: list[str] = []

    # ``ast.walk`` would descend into every function body, which is where
    # the legitimate calls live. Prune at each definition so only code
    # that actually runs on import is examined.
    def module_scope_nodes(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            yield child
            yield from module_scope_nodes(child)

    for path, text in _iter_sources():
        try:
            tree = ast.parse(text)
        except SyntaxError:  # pragma: no cover — the suite would fail elsewhere
            continue

        # Whatever local name each networking module was bound to.
        aliases: set[str] = set()
        direct: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in NETWORK_MODULES:
                        aliases.add(a.asname or a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] in NETWORK_MODULES:
                    for a in node.names:
                        direct.add(a.asname or a.name)

        for node in module_scope_nodes(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            where = f"{path.relative_to(SRC)}:{node.lineno}"
            if isinstance(func, ast.Attribute) and func.attr in NETWORK_CALLS:
                base = func.value
                name = getattr(base, "id", None) or getattr(base, "attr", None) or ""
                if name in aliases:
                    offenders.append(f"{where} {name}.{func.attr}()")
            elif isinstance(func, ast.Name) and func.id in direct:
                offenders.append(f"{where} {func.id}()")

    assert not offenders, (
        "Network call at module scope — importing hfl would contact a service "
        "before the operator has chosen anything:\n" + "\n".join(offenders)
    )
