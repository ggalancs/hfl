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

from fastapi import HTTPException, Request

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


def require_owner(request: Request, operation: str = "this operation") -> None:
    """Refuse ``operation`` for remote callers unless remote admin is on.

    Args:
        request: The incoming request (its peer address decides trust).
        operation: Human-readable operation name for the error message.

    Raises:
        HTTPException: ``403`` when the caller is remote and
            ``allow_remote_pull`` is not enabled.
    """
    if is_local_request(request):
        return

    # Late import so this module stays cheap and honours test monkeypatching
    # of ``hfl.config.config`` (mirrors ``routes_push._resolve_token``).
    from hfl.config import config

    if getattr(config, "allow_remote_pull", False):
        return

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
