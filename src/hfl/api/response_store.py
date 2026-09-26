# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Stored responses, for ``previous_response_id`` on ``/v1/responses``.

A client of OpenAI's Responses API can send only its new input and name
the response it follows; the server supplies the conversation before it.
Each stored response keeps its own turn — the input it was given (not its
``instructions``, which OpenAI does not carry over) and what the model
answered — and the response it followed, so a long conversation costs its
length once, not once per turn.

In memory, in this process: HFL is one process by design. The oldest
unused responses go first once ``max_turns`` are kept. A conversation
whose chain lost a link — evicted, or the server restarted — is reported
as not found, never continued from part of its history.
"""

from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from dataclasses import dataclass

from hfl.engine.base import ChatMessage

MAX_TURNS = 2048
MAX_DEPTH = 10_000  # a chain longer than this is refused, not walked


@dataclass(frozen=True)
class _Turn:
    parent: str | None
    messages: tuple[ChatMessage, ...]


class ResponseStore:
    def __init__(self, max_turns: int = MAX_TURNS) -> None:
        self._turns: OrderedDict[str, _Turn] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_turns

    def put(self, response_id: str, parent: str | None, messages: list[ChatMessage]) -> None:
        with self._lock:
            self._turns[response_id] = _Turn(parent, tuple(messages))
            self._turns.move_to_end(response_id)
            while len(self._turns) > self._max:
                self._turns.popitem(last=False)

    def history(self, response_id: str) -> list[ChatMessage] | None:
        """The conversation up to and including ``response_id``, oldest
        first; ``None`` when it, or any response before it, is not kept."""
        with self._lock:
            chain: list[_Turn] = []
            current: str | None = response_id
            while current is not None:
                turn = self._turns.get(current)
                if turn is None or len(chain) >= MAX_DEPTH:
                    return None
                self._turns.move_to_end(current)  # used: kept longer
                chain.append(turn)
                current = turn.parent
        # Copies: a caller building on the history must not edit what is kept.
        return [copy.deepcopy(message) for turn in reversed(chain) for message in turn.messages]

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()


_STORE = ResponseStore()


def get_response_store() -> ResponseStore:
    return _STORE
