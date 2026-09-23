# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Telling "the Hub is unreachable" apart from every other failure.

HFL's premise is that it runs locally and needs the network for exactly
one thing: fetching weights from the HuggingFace Hub. So being offline is
not an error condition for HFL, it is a normal one, and the Hub commands
have to say so plainly instead of surfacing whatever the socket layer
raised. Measured with DNS failing, before this module existed:

    hfl search llama   -> "Error searching: [Errno 8] nodename nor servname
                           provided, or not known"
    POST /api/pull     -> HTTP 500, as if the server itself had broken
    POST /api/pull/smart -> "the Hub reports no file sizes", which is false:
                           the Hub was never reached to report anything

Each of those is a message about a symptom. The cause — no network — is
one sentence, and the remedy is another: local models keep working.

Classification walks the exception chain, because huggingface_hub and
httpx both wrap the socket error they caught. It is deliberately
conservative about ``OSError``: ``FileNotFoundError`` is an ``OSError``
too, and calling a missing file "offline" would send the user chasing
their Wi-Fi. Only errnos that mean "the network did not answer" count.
"""

from __future__ import annotations

import errno
import logging
import socket

__all__ = ["HubUnreachableError", "is_network_error", "describe_hub_failure", "HUB_HOST"]

HUB_HOST = "huggingface.co"

# errnos that mean the packet never got an answer. ENOENT, EACCES, ENOSPC
# and friends are OSErrors as well and must stay out of this set.
_NETWORK_ERRNOS = frozenset(
    code
    for code in (
        getattr(errno, "ENETUNREACH", None),
        getattr(errno, "EHOSTUNREACH", None),
        getattr(errno, "ECONNREFUSED", None),
        getattr(errno, "ECONNRESET", None),
        getattr(errno, "ECONNABORTED", None),
        getattr(errno, "ETIMEDOUT", None),
        getattr(errno, "ENETDOWN", None),
        getattr(errno, "EHOSTDOWN", None),
    )
    if code is not None
)


class HubUnreachableError(ConnectionError):
    """The Hub could not be contacted at all.

    Raised where "could not reach" has to stay distinct from "reached, and
    the answer was empty" — the two call for opposite advice.
    """


def _is_network_link(exc: BaseException) -> bool:
    if isinstance(exc, HubUnreachableError):
        return True
    # gaierror / herror / timeout are OSError subclasses, so they go first.
    if isinstance(exc, (socket.gaierror, socket.herror, socket.timeout, TimeoutError)):
        return True
    if isinstance(exc, ConnectionError):  # refused, reset, aborted, broken pipe
        return True

    name = type(exc).__name__
    module = type(exc).__module__ or ""
    # httpx: TransportError covers connect/read timeouts and network errors,
    # and deliberately excludes HTTPStatusError — a 404 is an answer.
    if module.startswith("httpx") and name in {
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutException",
        "NetworkError",
        "ReadError",
        "WriteError",
        "RemoteProtocolError",
        "TransportError",
    }:
        return True
    # requests, still used by parts of the HF stack.
    if module.startswith("requests") and name in {"ConnectionError", "ConnectTimeout", "Timeout"}:
        return True
    # huggingface_hub's own "you asked for offline mode" signal.
    if name == "OfflineModeIsEnabled":
        return True

    if isinstance(exc, OSError) and exc.errno in _NETWORK_ERRNOS:
        return True
    return False


def is_network_error(exc: BaseException | None) -> bool:
    """Whether ``exc`` — or anything it was raised from — is a connectivity failure.

    Follows ``__cause__`` and ``__context__`` with a visited set, because
    wrapped errors can form loops and an unbounded walk would hang the very
    error path it is meant to clarify.
    """
    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _is_network_link(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def describe_hub_failure(log: logging.Logger, operation: str, exc: BaseException) -> str:
    """The message a caller gets when a Hub operation failed.

    Offline gets its own sentence, because it is the one failure the user
    can act on immediately and the one where local models still work. Any
    other failure keeps the reference-to-the-log treatment from
    ``log_internal_failure``, so no exception text reaches a client.
    """
    if is_network_error(exc):
        log.info("%s: Hub unreachable (%s)", operation, type(exc).__name__)
        return f"cannot reach {HUB_HOST} — this server appears to be offline"

    from hfl.logging_config import log_internal_failure

    return log_internal_failure(log, operation, exc)
