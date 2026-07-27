# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Regression tests for the 2026-07-27 security audit.

Every test here corresponds to a finding that was **verified as exploitable
before the fix** — not a hypothetical. The audit write-up lives in
``local/SECURITY_AUDIT_2026-07-27.md``.

A note that makes these tests work at all: ``TestClient`` presents
``client.host == "testclient"``, which is not in ``admin_guard._LOOPBACK_HOSTS``.
Every request from it is therefore evaluated as a **remote peer**, which is
exactly the caller the owner/user trust boundary is meant to refuse.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from hfl.api.server import APIKeyMiddleware, app
from hfl.api.state import get_state

client = TestClient(app)


# Every route that mutates server state, spends its disk, or changes what
# other consumers of the shared model receive. A new administrative route
# belongs in this list; the test then fails until it is guarded.
ADMIN_ROUTES: list[tuple[str, str, dict | None]] = [
    ("POST", "/api/pull", {"model": "org/repo"}),
    ("POST", "/api/pull/smart", {"model": "org/repo"}),
    ("POST", "/api/push", {"model": "m", "repo_id": "org/repo"}),
    ("POST", "/api/create", {"model": "probe", "from": "org/repo"}),
    ("POST", "/api/copy", {"source": "a", "destination": "b"}),
    ("POST", "/api/stop", {"model": "m"}),
    ("POST", "/api/lora/apply", {"model": "m", "lora_path": "a.gguf"}),
    ("POST", "/api/lora/remove", {"model": "m", "adapter_id": "ad1"}),
    ("POST", "/api/snapshot/save", {"model": "m", "name": "probe"}),
    ("POST", "/api/snapshot/load", {"model": "m", "name": "probe"}),
    ("DELETE", "/api/snapshot/probe", None),
    ("POST", "/api/batch", {"model": "m", "requests": [{"prompt": "hi"}]}),
]


class TestOwnerBoundary:
    """H-02. Before the fix, 8 of these 12 executed the handler for a remote
    peer: ``/api/stop`` returned 200 (evicting a 44 GiB model), ``/api/create``
    returned 200, and the rest reached their handler and failed only because
    the probe model did not exist."""

    @pytest.mark.parametrize("method,path,body", ADMIN_ROUTES, ids=[r[1] for r in ADMIN_ROUTES])
    def test_admin_route_refuses_remote_peer(self, method, path, body):
        resp = client.request(method, path, json=body) if body else client.request(method, path)
        assert resp.status_code == 403, (
            f"{method} {path} answered {resp.status_code} for a remote peer. "
            "Administrative routes must call require_owner() before doing work."
        )

    def test_guard_runs_before_the_body_is_used(self):
        """A schema-invalid body must not mask the authorization decision.

        During the audit an initial sweep sent wrong bodies and read the
        resulting 422s as 'protected'. They were not — validation simply ran
        first. This pins the property that a *valid* body still gets 403.
        """
        resp = client.post("/api/stop", json={"model": "m"})
        assert resp.status_code == 403


class TestCrossOriginAdmin:
    """M-05. The owner's browser is also a loopback peer, so 'local == owner'
    has a CSRF-shaped hole. CORS blocks the preflight but is a browser-side
    control: a request that does arrive is executed in full."""

    def test_foreign_origin_is_refused(self):
        resp = client.post(
            "/api/stop",
            json={"model": "m"},
            headers={"Origin": "https://evil.example"},
        )
        assert resp.status_code == 403
        assert resp.json()["detail"]["code"] == "cross_origin_admin_forbidden"

    def test_absent_origin_is_unaffected(self):
        """CLI / SDK / container clients never send Origin; they must keep
        being judged only by the peer address."""
        resp = client.post("/api/stop", json={"model": "m"})
        assert resp.json()["detail"]["code"] == "remote_admin_forbidden"


class TestErrorDisclosure:
    """H-01. The last-resort handler echoed ``str(exc)``. Verified with a
    synthetic exception carrying a path and a token-shaped string: both
    reached the client verbatim."""

    @staticmethod
    def _invoke(exc: Exception) -> dict:
        from hfl.api.exception_handlers import register_exception_handlers

        probe = FastAPI()
        register_exception_handlers(probe)
        handler = next(v for k, v in probe.exception_handlers.items() if k is Exception)
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/chat",
                "headers": [],
                "query_string": b"",
                "client": ("1.2.3.4", 1),
                "app": probe,
            }
        )
        loop = asyncio.new_event_loop()
        try:
            response = loop.run_until_complete(handler(request, exc))
        finally:
            loop.close()
        return json.loads(bytes(response.body).decode())

    def test_exception_text_is_not_returned(self):
        secret = "/Users/someone/.hfl/models/PRIVATE.gguf: token=hf_ABC123 line 42"
        body = self._invoke(RuntimeError(secret))
        blob = json.dumps(body)
        assert secret not in blob
        assert "hf_ABC123" not in blob
        assert ".gguf" not in blob

    def test_exception_class_name_is_not_returned(self):
        body = self._invoke(ZeroDivisionError("boom"))
        assert "ZeroDivisionError" not in json.dumps(body)
        assert "error_type" not in body

    def test_correlation_id_is_always_present(self):
        """The whole mitigation is 'find it in the log by request_id', so a
        null id would leave the operator with nothing."""
        body = self._invoke(RuntimeError("x"))
        assert body.get("request_id")


class TestPublicEndpointMatching:
    """M-04. Auth exemption was a ``startswith`` prefix test. uvicorn does not
    normalise ``..``, so ``/health/../api/chat`` skipped authentication; and
    ``/metricsfoo`` was exempt too."""

    @pytest.fixture(autouse=True)
    def with_api_key(self):
        get_state().api_key = "test-key"
        yield
        get_state().api_key = None

    @pytest.mark.parametrize(
        "path",
        [
            "/health/../api/tags",
            "/healthz-fake",
            "/metricsfoo",
            "/health-internal",
            "/metrics",
            "/metrics/json",
        ],
    )
    def test_lookalike_paths_require_the_key(self, path):
        assert client.get(path).status_code == 401

    @pytest.mark.parametrize("path", ["/health", "/healthz", "/api/version", "/"])
    def test_genuinely_public_paths_still_work(self, path):
        assert client.get(path).status_code != 401

    def test_metrics_can_be_opted_back_in(self, monkeypatch):
        monkeypatch.setenv("HFL_METRICS_PUBLIC", "true")
        assert client.get("/metrics").status_code == 200

    def test_public_set_is_exact_not_prefix(self):
        assert isinstance(APIKeyMiddleware.PUBLIC_ENDPOINTS, frozenset)
        assert not hasattr(APIKeyMiddleware, "PUBLIC_PREFIXES")


class TestAgentLoopGate:
    """H-03. ``agent_loop`` dispatches MCP tool calls server-side, and the
    request body both enables it and supplies the steering prompt."""

    def test_agent_loop_refused_by_default(self):
        resp = client.post(
            "/api/chat",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "agent_loop": True,
            },
        )
        assert resp.status_code == 403
        assert resp.json()["detail"]["code"] == "agent_loop_disabled"

    def test_gate_precedes_model_resolution(self):
        """403, not 404. Refusal must not depend on the model existing —
        otherwise the gate doubles as a model-existence oracle."""
        resp = client.post(
            "/api/chat",
            json={
                "model": "definitely-not-a-real-model-xyz",
                "messages": [{"role": "user", "content": "x"}],
                "agent_loop": True,
            },
        )
        assert resp.status_code == 403


class TestTemplateFieldResolution:
    """M-06. Templates are attacker-supplied (``template`` in a chat body,
    ``TEMPLATE`` in a Modelfile) and the resolved value is stringified into
    the prompt. ``getattr`` on arbitrary objects walked out of the data dict."""

    @staticmethod
    def _render(tpl: str):
        from hfl.converter.go_template import render_go_template

        return render_go_template(tpl, {"Prompt": "hello", "System": "", "Messages": []})

    @pytest.mark.parametrize(
        "tpl",
        [
            "{{ .Prompt.__class__ }}",
            "{{ .Prompt.__class__.__mro__ }}",
            "{{ .Messages.__class__.__doc__ }}",
            "{{ .Prompt.__class__.__init__.__globals__ }}",
            "{{ .System.__class__.__base__ }}",
        ],
    )
    def test_dunder_traversal_yields_nothing(self, tpl):
        assert self._render(tpl) == ""

    def test_legitimate_fields_still_render(self):
        assert self._render("{{ .Prompt }}") == "hello"
        assert self._render("a{{ .Missing }}b") == "ab"

    def test_attribute_access_on_objects_still_works(self):
        """Modelfile templates walk real objects (``{{ .Messages }}``), so the
        fix must not turn into a blanket ban on attribute access."""
        from hfl.converter.go_template import render_go_template

        class Turn:
            role = "user"

        assert render_go_template("{{ .Turn.role }}", {"Turn": Turn()}) == "user"


class TestSecurityHeaders:
    """L-14."""

    def test_baseline_headers_present(self):
        resp = client.get("/api/version")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["Referrer-Policy"] == "no-referrer"


class TestPublicBindDetection:
    """M-09. The exposure warning only recognised the literal ``0.0.0.0``."""

    @pytest.mark.parametrize(
        "host,public",
        [
            ("0.0.0.0", True),  # noqa: S104 - test data, not a bind
            ("::", True),
            ("192.168.1.10", True),
            ("10.0.0.5", True),
            ("example.com", True),  # unresolvable → assume exposed
            ("", True),
            ("127.0.0.1", False),
            ("::1", False),
        ],
    )
    def test_detects_every_public_bind(self, host, public):
        from hfl.cli.main import _is_public_bind

        assert _is_public_bind(host) is public


class TestAuthBackoff:
    """L-13."""

    def test_repeated_failures_are_counted(self):
        from hfl.api import server as srv

        srv._AUTH_FAILURES.clear()
        get_state().api_key = "test-key"
        try:
            for _ in range(3):
                assert client.get("/api/tags", headers={"X-API-Key": "wrong"}).status_code == 401
            assert sum(srv._AUTH_FAILURES.values()) >= 3
            # A correct key clears the peer's counter.
            client.get("/api/tags", headers={"X-API-Key": "test-key"})
            assert sum(srv._AUTH_FAILURES.values()) == 0
        finally:
            get_state().api_key = None
            srv._AUTH_FAILURES.clear()


class TestPathMatchingIsExactEverywhere:
    """Second-audit finding: the ``startswith`` path-matching bug fixed in
    ``APIKeyMiddleware`` also existed in the body-limit and rate-limit
    middlewares. Fixing one instance of a pattern is not fixing the pattern.
    """

    def test_body_limit_exemption_is_exact(self):
        from hfl.api.middleware import RequestBodyLimitMiddleware as M

        assert isinstance(M.EXCLUDED_PATHS, frozenset)
        assert not hasattr(M, "EXCLUDED_PREFIXES")
        assert "/api/transcribe" in M.EXCLUDED_PATHS
        # A look-alike must NOT inherit the exemption.
        assert "/api/transcribe-evil" not in M.EXCLUDED_PATHS

    def test_rate_limit_exemption_is_exact(self):
        from hfl.api.middleware import RateLimitMiddleware as M

        assert isinstance(M.EXCLUDED_PATHS, frozenset)
        assert not hasattr(M, "EXCLUDED_PREFIXES")
        assert "/health" in M.EXCLUDED_PATHS
        assert "/health-anything" not in M.EXCLUDED_PATHS


class TestTranscribeUploadBound:
    """Second-audit finding: the route is exempt from the global body limit
    and checked the size only *after* an unbounded ``file.read()``."""

    def test_read_is_bounded(self):
        import inspect

        from hfl.api import routes_transcribe

        src = inspect.getsource(routes_transcribe)
        assert "await file.read(_MAX_AUDIO_BYTES + 1)" in src, (
            "the upload must be read with an explicit bound, otherwise the "
            "413 fires only after the whole body has been buffered"
        )
        assert "await file.read()\n" not in src


class TestLogRedaction:
    """Third-audit finding: HFL's own request logger records only
    ``request.url.path``, but uvicorn's access log records the path *with*
    its query string and is enabled by default — so a WebSocket handshake
    carrying ``?api_key=…`` (the only form a browser can use) wrote the key
    into the log file. Fixing the header/query preference order in
    ``routes_ws`` did not close that channel; it lives outside HFL's code.
    """

    @staticmethod
    def _redact(message: str) -> str:
        import logging

        from hfl.logging_config import SensitiveQueryFilter

        record = logging.LogRecord("uvicorn.access", 20, "", 0, message, None, None)
        SensitiveQueryFilter().filter(record)
        return str(record.msg)

    @pytest.mark.parametrize("param", ["api_key", "token", "access_token", "password", "API_KEY"])
    def test_sensitive_params_are_redacted(self, param):
        out = self._redact(f'GET /ws/chat?{param}=SUPERSECRET HTTP/1.1" 200')
        assert "SUPERSECRET" not in out
        assert "=***" in out

    def test_only_the_value_is_removed(self):
        out = self._redact("GET /ws/chat?foo=1&api_key=abc123&bar=2 HTTP/1.1")
        assert "abc123" not in out
        assert "foo=1" in out and "bar=2" in out  # the rest stays useful

    def test_ordinary_lines_are_untouched(self):
        line = 'GET /api/tags HTTP/1.1" 200 OK'
        assert self._redact(line) == line

    def test_filter_is_installed_on_uvicorn_access(self):
        import logging

        from hfl.logging_config import SensitiveQueryFilter, configure_logging

        configure_logging(level="INFO")
        filters = logging.getLogger("uvicorn.access").filters
        assert any(isinstance(f, SensitiveQueryFilter) for f in filters)
