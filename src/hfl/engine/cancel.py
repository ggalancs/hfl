# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The cancellation signal of the request an engine call serves.

``run_dispatched`` gives every inference a fresh ``threading.Event`` and sets
it when the request runs past its budget (a 504) or is abandoned. The event
travels in a ContextVar — ``asyncio.to_thread`` copies the context into the
worker thread — so each call sees *its own* request's signal. That matters
for the engines that serve several requests at once (llama-server, vLLM): an
engine-wide "stop" would cut whichever request happened to be running.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_current: ContextVar[threading.Event | None] = ContextVar("hfl_cancel", default=None)


class GenerationCancelled(RuntimeError):
    """The request this generation served was cancelled; its partial result
    was not wanted."""


def current() -> threading.Event | None:
    """The running request's signal, or None outside a dispatched call."""
    return _current.get()


def cancelled() -> bool:
    event = _current.get()
    return event is not None and event.is_set()


@contextmanager
def scope(event: threading.Event) -> Iterator[threading.Event]:
    """Make ``event`` the signal of calls started inside."""
    token = _current.set(event)
    try:
        yield event
    finally:
        _current.reset(token)
