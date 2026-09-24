# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Owner-vs-user trust boundary for administrative API endpoints.

HFL is designed to run on the operator's own machine. When it is
exposed over the network to serve inference to *users*, some endpoints
remain **owner** operations that a remote user must not be able to
trigger: ``pull``, smart-pull and ``push`` download or upload arbitrary
repositories on the server host, consume its disk, and implicitly
"accept" model licenses that are not the caller's to accept.

The rule this module enforces: those endpoints are allowed only for a
**local (loopback) caller** — i.e. the owner working on the box — unless
the owner has explicitly opted into remote administration via
``HFL_ALLOW_REMOTE_PULL=true`` (in which case the API key still guards
them). Remote users get a clean ``403`` and are steered toward local
provisioning.

This is the same posture Ollama takes by binding to ``127.0.0.1`` by
default: exposing the box is a deliberate act with the owner's
responsibility attached.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

# Loopback peers are the machine's owner. Everything else is a remote
# *user*. ``localhost`` is included for transports that pass the name
# through unresolved.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def is_local_request(request: Request) -> bool:
    """Whether the request originates from the loopback interface.

    Fails safe: if the ASGI transport reports no peer (``request.client``
    is ``None``), the caller is treated as **remote** so the guard errs
    toward refusal rather than exposure.
    """
    client = request.client
    if client is None:
        return False
    return client.host in _LOOPBACK_HOSTS


def _reject_browser_origin(request: Request, operation: str) -> None:
    """Refuse an administrative call that carries a cross-origin ``Origin``.

    "Loopback peer == the owner" is the premise of this module, and it has
    one hole: **the owner's browser is also a loopback peer**. Any page the
    owner visits can issue requests to ``http://127.0.0.1:11434`` and they
    arrive looking exactly like the owner working locally. CORS is what
    stops this today — the default allow-list is empty, so a browser's
    preflight for a JSON POST fails and the request is never sent — but
    CORS is a *browser* control, not an authorization control: verified in
    audit, a request that does reach the server is executed in full and
    only the response is hidden from the page. And the project documents
    ``cors_allow_all=True`` as a development option, which removes the
    protection entirely.

    So: a non-browser caller (CLI, container, SDK) never sends ``Origin``.
    A cross-origin browser always does. Refusing administrative operations
    that carry a foreign ``Origin`` closes the CSRF path without breaking
    any legitimate client.
    """
    origin = request.headers.get("origin")
    if not origin:
        return  # CLI / SDK / container — no browser involved.

    from hfl.config import config

    if config.cors_allow_all:
        return
    if origin in (config.cors_origins or []):
        return

    raise HTTPException(
        status_code=403,
        detail={
            "error": (
                f"{operation} is an owner (administrative) operation and cannot be "
                f"triggered from a web page (Origin: {origin}). Use the CLI or an "
                "HTTP client, or add the origin to HFL_ORIGINS if you trust it."
            ),
            "code": "cross_origin_admin_forbidden",
            "category": "auth",
            "retryable": False,
        },
    )


# ``require_owner``'s operation strings are written for a human reading a
# 403. The audit log is read by a machine, so the two vocabularies are
# mapped explicitly rather than derived — a derivation would silently
# invent an event name the catalogue does not know.
_AUDIT_EVENT_FOR: dict[str, str] = {
    "pull": "model.pull",
    "smart-pull": "model.smart_pull",
    "push": "model.push",
    "create": "model.create",
    "copy": "model.copy",
    "stop": "model.stop",
    "batch": "model.batch",
    "lora apply": "lora.apply",
    "lora remove": "lora.remove",
    "snapshot save": "snapshot.save",
    "snapshot load": "snapshot.load",
    "snapshot delete": "snapshot.delete",
}


def _actor_for(request: Request) -> str:
    """Identify the caller without ever recording a credential.

    A loopback peer is the machine's owner and is recorded as such. A
    remote peer is identified by a short PBKDF2 digest of its API key
    salted with a random per-process secret: enough to correlate a series of actions to
    one client within a server run, and useless to anybody who obtains the
    log. A plain hash was not: its prefix let a reader of the log test
    guesses of a weak, human-chosen key offline (CodeQL
    py/weak-sensitive-data-hashing). The price is that the same client
    gets a different label after a restart.
    """
    if is_local_request(request):
        return "local"
    key = request.headers.get("authorization", "") or request.headers.get("x-api-key", "")
    if key.lower().startswith("bearer "):
        key = key[7:]
    if not key:
        return "anonymous"
    return "api-key:" + _actor_digest(key)[:8]


# Never persisted or logged: it exists so actor labels cannot be reversed.
_ACTOR_SECRET = os.urandom(32)


@functools.lru_cache(maxsize=64)
def _actor_digest(key: str) -> str:
    """PBKDF2 keyed by the process secret: slow to guess against and
    impossible without the secret. Cached, so a client pays it once."""
    return hashlib.pbkdf2_hmac("sha256", key.encode("utf-8"), _ACTOR_SECRET, 20_000).hex()


def _audit(request: Request, operation: str, outcome: str) -> None:
    """Record a privileged attempt. Never raises, never blocks the route.

    Denied attempts are recorded too, and are the more interesting half:
    an audit log that only holds successes cannot answer the question it
    exists for.
    """
    event = _AUDIT_EVENT_FOR.get(operation)
    if event is None:
        return
    try:
        from hfl.observability.audit import audit_event

        audit_event(event, actor=_actor_for(request), outcome=outcome)
    except Exception:  # pragma: no cover — auditing must never break a route
        logger.debug("audit emit failed for %s", event, exc_info=True)


def require_owner(request: Request, operation: str = "this operation") -> None:
    """Refuse ``operation`` for remote callers unless remote admin is on.

    Args:
        request: The incoming request (its peer address decides trust).
        operation: Human-readable operation name for the error message.

    Raises:
        HTTPException: ``403`` when the caller is remote and
            ``allow_remote_pull`` is not enabled, or when the call carries
            a cross-origin ``Origin`` header (see
            :func:`_reject_browser_origin`).
    """
    # Always first: a browser on the loopback interface passes the peer
    # test below, so the Origin check has to run for local callers too.
    _reject_browser_origin(request, operation)

    if is_local_request(request):
        _audit(request, operation, "ok")
        return

    # Late import so this module stays cheap and honours test monkeypatching
    # of ``hfl.config.config`` (mirrors ``routes_push._resolve_token``).
    from hfl.config import config

    if getattr(config, "allow_remote_pull", False):
        _audit(request, operation, "ok")
        return

    _audit(request, operation, "denied")
    raise HTTPException(
        status_code=403,
        detail={
            "error": (
                f"{operation} is an owner (administrative) operation and cannot be "
                "triggered by a remote API client. Provision models locally on the "
                "server host, or set HFL_ALLOW_REMOTE_PULL=true if you knowingly "
                "administer this server remotely."
            ),
            "code": "remote_admin_forbidden",
            "category": "auth",
            "retryable": False,
        },
    )


def require_local_owner(request: Request, operation: str) -> None:
    """Like :func:`require_owner`, with no remote exception at all.

    For operations that destroy data on the host (deleting a model):
    ``HFL_ALLOW_REMOTE_PULL`` opens provisioning to remote clients, not
    destruction. Only a loopback peer, and never from a web page.
    """
    _reject_browser_origin(request, operation)
    if is_local_request(request):
        _audit(request, operation, "ok")
        return
    _audit(request, operation, "denied")
    raise HTTPException(
        status_code=403,
        detail={
            "error": (
                f"{operation} can only be done on the server host itself "
                "(loopback), never by a remote API client."
            ),
            "code": "local_owner_only",
            "category": "auth",
            "retryable": False,
        },
    )
