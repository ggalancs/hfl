# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The audit log, with something actually writing to it.

`observability/audit.py` shipped a complete append-only JSONL sink with
rotation, an event catalogue, and a docstring promising "every privileged
route / CLI command that mutates server state records exactly one event".
Nothing called `audit_event`. The log existed and was always empty, which
is worse than not having one: an empty audit log reads as "nothing
happened".

The emitter is a single call site inside `require_owner`, because that is
the one gate every privileged route already passes through. Twelve
`audit_event(...)` calls sprinkled across twelve routers would drift the
first time somebody adds the thirteenth; one gate cannot.

Denied attempts are recorded as well as allowed ones, and are the more
interesting half — a log that only holds successes cannot answer the
question an audit log exists for.
"""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers

from hfl.api.admin_guard import _AUDIT_EVENT_FOR, _actor_for, require_owner
from hfl.observability.audit import AUDIT_EVENTS, configure_audit_log, reset_audit_log

LOCAL = ("127.0.0.1", 5555)
REMOTE = ("203.0.113.7", 5555)


class _Req:
    """Minimal stand-in: require_owner reads only the peer and the headers."""

    def __init__(self, peer, headers=None):
        self.client = type("C", (), {"host": peer[0], "port": peer[1]})()
        self.headers = Headers(headers or {})


@pytest.fixture
def audit_file(tmp_path):
    path = tmp_path / "audit.jsonl"
    reset_audit_log()
    configure_audit_log(path)
    yield path
    reset_audit_log()


def _events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class TestSomethingIsFinallyWritten:
    def test_an_allowed_privileged_call_is_recorded(self, audit_file):
        require_owner(_Req(LOCAL), "pull")
        events = _events(audit_file)
        assert len(events) == 1, "the audit log is still empty"
        assert events[0]["event"] == "model.pull"
        assert events[0]["outcome"] == "ok"

    def test_a_denied_call_is_recorded_too(self, audit_file, monkeypatch):
        """The half that matters most."""
        import hfl.config

        monkeypatch.setattr(hfl.config.config, "allow_remote_pull", False)
        with pytest.raises(HTTPException):
            require_owner(_Req(REMOTE), "push")

        events = _events(audit_file)
        assert [e["outcome"] for e in events] == ["denied"]
        assert events[0]["event"] == "model.push"

    def test_exactly_one_event_per_call(self, audit_file):
        require_owner(_Req(LOCAL), "create")
        require_owner(_Req(LOCAL), "create")
        assert len(_events(audit_file)) == 2


class TestNoCredentialEverReachesTheLog:
    def test_a_remote_actor_is_a_hash_not_the_key(self):
        secret = "sk-super-secret-value"
        actor = _actor_for(_Req(REMOTE, {"authorization": f"Bearer {secret}"}))
        assert secret not in actor
        assert actor.startswith("api-key:")
        assert len(actor) == len("api-key:") + 8

    def test_the_same_key_always_hashes_the_same(self):
        """Correlation is the point; a random id per request would be useless."""
        one = _actor_for(_Req(REMOTE, {"authorization": "Bearer abc"}))
        two = _actor_for(_Req(REMOTE, {"authorization": "Bearer abc"}))
        assert one == two

    def test_different_keys_differ(self):
        assert _actor_for(_Req(REMOTE, {"authorization": "Bearer a"})) != _actor_for(
            _Req(REMOTE, {"authorization": "Bearer b"})
        )

    def test_a_local_caller_is_named_local(self):
        assert _actor_for(_Req(LOCAL)) == "local"

    def test_the_label_is_keyed_so_a_log_reader_cannot_test_guesses(self):
        """Anyone can compute sha256("letmein")[:8]; nobody outside the
        process can compute its HMAC under the process secret."""
        import hashlib

        actor = _actor_for(_Req(REMOTE, {"authorization": "Bearer letmein"}))
        assert actor != "api-key:" + hashlib.sha256(b"letmein").hexdigest()[:8]

    def test_the_key_never_appears_in_the_written_record(self, audit_file, monkeypatch):
        import hfl.config

        monkeypatch.setattr(hfl.config.config, "allow_remote_pull", True)
        require_owner(_Req(REMOTE, {"authorization": "Bearer leak-me-please"}), "stop")
        assert "leak-me-please" not in audit_file.read_text(encoding="utf-8")


class TestTheCatalogueStaysHonest:
    def test_every_mapped_event_is_in_the_catalogue(self):
        """`audit_event` only warns on an unknown name, and a warning in a
        log nobody reads is how an event silently stops being recorded."""
        unknown = sorted(set(_AUDIT_EVENT_FOR.values()) - AUDIT_EVENTS)
        assert not unknown, f"emitted but not in AUDIT_EVENTS: {unknown}"

    def test_every_guarded_operation_has_an_event(self):
        """Adding a privileged route without an audit name fails here.

        Scans the routers for the operation strings actually passed to
        `require_owner`, so a thirteenth admin route cannot arrive
        unaudited.
        """
        import re
        from pathlib import Path

        import hfl.api as api_pkg

        pattern = re.compile(r"require_owner\(\s*request\s*,\s*\"([^\"]+)\"")
        found: set[str] = set()
        for route in Path(api_pkg.__file__).parent.glob("routes_*.py"):
            found.update(pattern.findall(route.read_text(encoding="utf-8")))

        assert found, "no require_owner call sites found — has the guard moved?"
        unmapped = sorted(found - set(_AUDIT_EVENT_FOR))
        assert not unmapped, (
            f"privileged operations with no audit event: {unmapped}. Add them to "
            "_AUDIT_EVENT_FOR and to AUDIT_EVENTS, or they mutate the server "
            "without leaving a record."
        )


class TestAuditingNeverBreaksTheRoute:
    def test_a_failing_sink_does_not_stop_a_privileged_call(self, monkeypatch):
        """The audit path must never abort the operation it observes."""
        import hfl.observability.audit as audit_mod

        def _boom(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(audit_mod, "audit_event", _boom)
        require_owner(_Req(LOCAL), "pull")  # must not raise

    def test_an_unmapped_operation_is_skipped_quietly(self, audit_file):
        """`require_owner` has a default operation string used by nothing
        privileged; it must not produce a bogus event."""
        require_owner(_Req(LOCAL), "this operation")
        assert _events(audit_file) == []
