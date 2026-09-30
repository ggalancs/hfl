# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""One streamed chat turn, for every API dialect.

The four chat surfaces (Ollama ``/api/chat``, OpenAI ``/v1/chat/completions``
and ``/v1/responses``, Anthropic ``/v1/messages``) each carried the same
machinery around their wire format: the not-loaded answer, the engine's
stream (with or without tools), the reasoning splitter, the accumulated
text, the timings and token counts, backpressure, an error that never
leaks internals, and the dispatcher slot released whatever happens. A fix
made in one had to be made in four. :func:`run_chat_stream` is that
machinery; a dialect is a :class:`Renderer` that only formats.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Protocol

if TYPE_CHECKING:
    from hfl.api.chat_core import ChatOutput
    from hfl.engine.base import ChatMessage, GenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """A streamed turn as a renderer sees it."""

    model: str
    config: GenerationConfig
    tools: list[dict] | None
    stream: Iterator[str] = field(default_factory=lambda: iter(()))
    accumulated: list[str] = field(default_factory=list)
    emitted: int = 0
    start_ns: int = field(default_factory=time.monotonic_ns)
    first_token_ns: int | None = None
    splitter: Any = None

    def __post_init__(self) -> None:
        from hfl.api.thinking import ThinkingSplitter

        self.splitter = ThinkingSplitter()

    @property
    def tool_aware(self) -> bool:
        """Tools declared: a tool-call marker must never stream as text."""
        return bool(self.tools)

    def text(self) -> str:
        return "".join(self.accumulated)

    def counts(self) -> tuple[int | None, int | None]:
        """(prompt, generated) as the engine counted them, None where it
        does not count."""
        from hfl.engine.base import stream_counts

        return stream_counts(self.stream)

    def generated(self) -> int:
        """Tokens generated: the engine's count, else the chunks emitted."""
        generated = self.counts()[1]
        return generated if generated is not None else self.emitted

    def hit_cap(self) -> bool:
        """Whether generation stopped at ``max_tokens``."""
        return bool(self.config.max_tokens) and self.generated() >= self.config.max_tokens

    def resolved(self) -> ChatOutput:
        """The whole reply as content, tool calls and reasoning."""
        from hfl.api.chat_core import resolve_chat_output

        return resolve_chat_output(self.text(), self.model, self.tools)


class Renderer(Protocol):
    """A dialect's wire format for a streamed turn."""

    log_label: str

    def not_loaded(self) -> str: ...

    def failed(self) -> str: ...

    def start(self, turn: Turn) -> str: ...

    def token(self, turn: Turn, token: str) -> str: ...

    def done(self, turn: Turn) -> str: ...


def _engine_stream(engine: Any, messages: list[ChatMessage], config: Any, tools: Any) -> Any:
    """The engine's token stream; an engine without ``tools`` support gets
    the two-argument call."""
    if tools is None:
        return engine.chat_stream(messages, config)
    try:
        return engine.chat_stream(messages, config, tools=tools)
    except TypeError:
        return engine.chat_stream(messages, config)


async def run_chat_stream(
    renderer: Renderer,
    *,
    engine: Any,
    model: str,
    messages: list[ChatMessage],
    config: GenerationConfig,
    tools: list[dict] | None = None,
    slot_cm: Any | None = None,
) -> AsyncIterator[str]:
    """Stream one chat turn through ``renderer``, with backpressure.

    ``engine`` is the route's (``state.engine``, read through the route's
    own accessor). ``slot_cm`` is the dispatcher slot held for the whole
    stream (spec §5.3); it is released in ``finally``, whatever happens.
    """
    from hfl.api.streaming import stream_with_backpressure

    try:
        if engine is None:
            yield renderer.not_loaded()
            return
        turn = Turn(model=model, config=config, tools=tools)
        turn.stream = _engine_stream(engine, messages, config, tools)
        preamble = renderer.start(turn)
        if preamble:
            yield preamble

        def format_item(token: str) -> str:
            turn.accumulated.append(token)
            turn.emitted += 1
            if turn.first_token_ns is None:
                turn.first_token_ns = time.monotonic_ns()
            return renderer.token(turn, token)

        try:
            async for chunk in stream_with_backpressure(
                sync_iterator=turn.stream,
                format_item=format_item,
                format_done=lambda: renderer.done(turn),
            ):
                yield chunk
        except Exception:
            # Never ``str(exc)`` on the stream: it can reveal paths, library
            # classes or line numbers (CodeQL py/stack-trace-exposure). The
            # traceback goes to the server log.
            logger.exception("%s stream failed for model %s", renderer.log_label, model)
            yield renderer.failed()
    finally:
        if slot_cm is not None:
            with contextlib.suppress(Exception):
                await slot_cm.__aexit__(None, None, None)
