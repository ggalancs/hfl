# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""V4 F7 — ``WS /ws/chat`` bidirectional chat.

The streaming chat endpoints today are HTTP one-shots: client opens
a request, server streams tokens, connection closes. Cancellation
relies on TCP close — fine, but it doesn't carry an actionable
signal back ("I cancelled because the user clicked stop, you can
reuse this slot for the next prompt").

The WebSocket endpoint adds:

- A persistent connection where the client can submit multiple
  ``chat`` messages without re-opening.
- A ``cancel`` frame that interrupts the in-flight generation
  cleanly, releases the dispatcher slot, and leaves the connection
  open for the next prompt.
- Server-emitted ``token`` / ``done`` / ``error`` frames so the
  client can render token-level UI without parsing NDJSON or SSE.

Frame grammar (JSON, one frame per WebSocket message):

  client → server:
    { "type": "chat", "model": "...", "messages": [...], "options": {...}? }
    { "type": "cancel" }                  # cancels the current chat
    { "type": "ping" }                    # heartbeat

  server → client:
    { "type": "ready", "model": "..." }   # sent once after chat is accepted
    { "type": "token", "delta": "..." }   # one per generated chunk
    { "type": "done", "tokens": N }       # sent once at end of generation
    { "type": "error", "message": "..." } # any failure
    { "type": "pong" }                    # ping reply
    { "type": "cancelled" }               # confirmation of a cancel
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from hfl.exceptions import HFLError
from hfl.logging_config import log_internal_failure

logger = logging.getLogger(__name__)

router = APIRouter(tags=["HFL Beyond"])


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------


async def _send(ws: WebSocket, frame: dict[str, Any]) -> None:
    """Send a JSON frame; swallow disconnects so the caller's
    finally block can keep running cleanup without surfacing the
    error."""
    try:
        await ws.send_text(json.dumps(frame))
    except (WebSocketDisconnect, RuntimeError):
        pass


def _validate_chat_frame(frame: dict[str, Any]) -> tuple[str, list[dict]]:
    """Pull ``model`` and ``messages`` from a ``chat`` frame.

    Raises ``ValueError`` with a human message when the frame is
    malformed — the route turns that into an ``error`` frame
    without disconnecting.
    """
    model = frame.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("'model' must be a non-empty string")
    messages = frame.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("'messages' must be a non-empty list")
    for m in messages:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError("each message needs 'role' and 'content'")
    return model.strip(), messages


# ---------------------------------------------------------------------------
# Generation driver
# ---------------------------------------------------------------------------


class _SocketGeneration:
    """The one generation a socket may have running.

    A ``cancel`` ends the *turn* at once, but not the engine call: a sync
    backend runs on in its thread until it returns. A new turn that started
    beside it was a second engine call per chat→cancel round, outside every
    bound — one socket could fill the thread pool. ``running`` is the
    background task that ends only when that call has returned and its
    dispatcher slot is free; the next turn waits for it."""

    def __init__(self) -> None:
        self.running: asyncio.Future[Any] | None = None


# Strong references to slot-releasing tasks: the event loop keeps only weak
# ones, and the turn that started a task may be long gone when it finishes.
_releasers: set[asyncio.Task[None]] = set()


async def _abandon_acquire(acquire: "asyncio.Future[Any]", slot_cm: Any) -> None:
    """Withdraw a pending dispatcher-slot request; release the slot if the
    request won the race and got one anyway."""
    acquire.cancel()
    try:
        await acquire
    except BaseException:
        return  # never acquired (cancelled, or refused)
    with contextlib.suppress(Exception):
        await slot_cm.__aexit__(None, None, None)


def _max_tokens(options: dict[str, Any]) -> int | None:
    """``options.max_tokens`` when the client sent one, else ``None`` (the
    GenerationConfig default, the same bounded one HTTP gets)."""
    value = options.get("max_tokens")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("'max_tokens' must be a positive integer")
    return value


def _log_drive_result(ws: WebSocket, task: "asyncio.Task[None]") -> None:
    """Retrieve a background chat task's exception so it is logged rather than
    surfacing as an unretrieved-task warning at GC.

    ``_drive_chat`` surfaces expected failures (bad frame, load error, engine
    error) as ``error`` frames itself, so reaching here means an *unexpected*
    post-``ready`` failure. Emit a terminal ``error`` frame so the client is
    not left waiting for a turn that will never complete — the receive loop
    stays open for the next prompt (previously such an error tore the whole
    connection down)."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("ws chat turn failed: %s", exc, exc_info=exc)
        with contextlib.suppress(Exception):
            asyncio.get_running_loop().create_task(
                _send(ws, {"type": "error", "message": "internal error during generation"})
            )


async def _drive_chat(
    ws: WebSocket,
    frame: dict[str, Any],
    cancel_event: asyncio.Event,
    generation: _SocketGeneration | None = None,
) -> None:
    """Run one chat turn over the WebSocket.

    Cancellation contract: a ``cancel`` frame from the client sets
    ``cancel_event`` (owned by the receive loop, which keeps reading frames
    while this turn runs in a background task); the driver races it against
    each produced token and exits cleanly when it fires.
    """
    if generation is None:
        generation = _SocketGeneration()
    try:
        model_name, messages = _validate_chat_frame(frame)
        options = frame.get("options") or {}
        if not isinstance(options, dict):
            raise ValueError("'options' must be an object")
        max_tokens = _max_tokens(options)
    except ValueError as exc:
        await _send(ws, {"type": "error", "message": str(exc)})
        return

    # SEC: one generation per socket. The previous turn's engine call may
    # still be running after its cancel; this turn starts when it has
    # returned — or ends here if it is cancelled first.
    previous = generation.running
    if previous is not None and not previous.done():
        waiter = asyncio.ensure_future(asyncio.shield(previous))
        cancelled_wait = asyncio.ensure_future(cancel_event.wait())
        try:
            await asyncio.wait({waiter, cancelled_wait}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
            cancelled_wait.cancel()
        if cancel_event.is_set():
            await _send(ws, {"type": "cancelled", "tokens": 0})
            return

    from hfl.api.model_loader import load_llm
    from hfl.engine.base import ChatMessage, GenerationConfig

    try:
        engine, manifest = await load_llm(model_name)
    except HFLError as exc:
        # HFL's own errors carry a message we wrote for the caller
        # (ModelNotFoundError, ModelTypeMismatchError, ...).
        await _send(ws, {"type": "error", "message": str(exc)})
        return
    except Exception as exc:
        detail = log_internal_failure(logger, "load_llm", exc)
        await _send(ws, {"type": "error", "message": detail})
        return

    if engine is None:
        await _send(ws, {"type": "error", "message": "engine not available"})
        return

    # CON: the WS turn reads ``engine`` directly, not through
    # ``run_dispatched``. With several models resident a concurrent load may
    # choose this one to evict, and an explicit unload could free it
    # mid-stream (use-after-free of the non-reentrant model). Lease it before
    # the first await, so no
    # eviction can land between the load and the lease. From the producer's
    # start the release belongs to the producer (it can outlive a client
    # cancel — the engine call cannot be preempted); anything failing before
    # that releases here.
    from hfl.api.state import get_state

    state = get_state()
    await state.pin_engine(engine)
    try:
        chat_msgs = [
            ChatMessage(role=str(m.get("role")), content=str(m.get("content") or ""))
            for m in messages
        ]
        cfg = GenerationConfig(
            temperature=float(options.get("temperature", 0.7) or 0.7),
            top_p=float(options.get("top_p", 0.9) or 0.9),
        )
        # SEC: an omitted max_tokens keeps the bounded default HTTP uses. It
        # was 0 here — "until the context is full" — for every turn.
        if max_tokens is not None:
            cfg.max_tokens = max_tokens
        # A created model's Modelfile (SYSTEM, MESSAGE, PARAMETER defaults).
        from hfl.api.modelfile_defaults import apply_to_chat

        sent = {"max_tokens": "num_predict", "temperature": "temperature", "top_p": "top_p"}
        explicit = {name for key, name in sent.items() if options.get(key) is not None}
        chat_msgs = apply_to_chat(manifest, chat_msgs, cfg, explicit)
    except BaseException:
        await state.unpin_engine(engine)
        raise

    # SEC: the same dispatcher slot HTTP inference takes. Outside it, a
    # turn ran beside whatever the model was serving (a second call on a
    # non-reentrant model) and no queue bound or 429 ever applied to it.
    from hfl.api.helpers import _cancel_engine, suggest_parallel
    from hfl.core import dispatcher_for
    from hfl.engine import cancel as _cancel
    from hfl.engine.dispatcher import QueueFullError, QueueTimeoutError

    dispatcher = dispatcher_for(engine)
    suggest_parallel(dispatcher, engine)
    slot_cm = dispatcher.slot()
    # Raced against ``cancel``: a turn the client gave up on while queued
    # must leave the queue, not run once a slot frees.
    acquire = asyncio.ensure_future(slot_cm.__aenter__())
    stop = asyncio.ensure_future(cancel_event.wait())
    try:
        await asyncio.wait({acquire, stop}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        stop.cancel()
        await _abandon_acquire(acquire, slot_cm)
        await state.unpin_engine(engine)
        raise
    stop.cancel()
    if not acquire.done():
        await _abandon_acquire(acquire, slot_cm)
        await state.unpin_engine(engine)
        await _send(ws, {"type": "cancelled", "tokens": 0})
        return
    try:
        acquire.result()
    except QueueFullError as exc:
        await state.unpin_engine(engine)
        await _send(
            ws,
            {
                "type": "error",
                "message": "server busy: inference queue full",
                "code": "queue_full",
                "retry_after": exc.retry_after_seconds,
            },
        )
        return
    except QueueTimeoutError:
        await state.unpin_engine(engine)
        await _send(
            ws,
            {
                "type": "error",
                "message": "timed out waiting for the model",
                "code": "queue_timeout",
            },
        )
        return
    except BaseException:
        await state.unpin_engine(engine)
        raise

    try:
        await _send(ws, {"type": "ready", "model": model_name})
    except BaseException:
        await slot_cm.__aexit__(None, None, None)
        await state.unpin_engine(engine)
        raise

    # Run the sync chat_stream in a thread so the event loop can
    # service inbound ``cancel`` frames concurrently.
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _producer() -> None:
        # ``asyncio.Queue`` is NOT thread-safe, and this closure runs in a
        # worker thread. Every enqueue must be marshalled back onto the loop
        # via ``call_soon_threadsafe`` (which also wakes the loop's selector).
        # Calling ``put_nowait`` directly from the thread races the loop's own
        # access to the queue internals and can drop the consumer's getter
        # wakeup, stalling the turn — mirror the streaming.py pattern. (CON)
        try:
            for token in engine.chat_stream(chat_msgs, cfg):
                loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
        except Exception as exc:  # pragma: no cover — surface as error frame
            # Backend exceptions (llama.cpp, torch) name paths and internals.
            # The socket gets the reference, the log gets the traceback — the
            # same posture the OpenAI / Ollama streaming paths already take.
            detail = log_internal_failure(logger, "chat stream", exc)
            loop.call_soon_threadsafe(queue.put_nowait, ("error", detail))
        finally:
            # The engine is no longer being read — release the pin so a
            # displaced engine can finally be unloaded.
            asyncio.run_coroutine_threadsafe(state.unpin_engine(engine), loop)
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    # This turn's own cancellation signal, seen by the worker thread
    # (to_thread copies the context), as ``run_dispatched`` does.
    signal = threading.Event()
    with _cancel.scope(signal):
        producer_task = asyncio.ensure_future(asyncio.to_thread(_producer))

    async def _release_when_producer_exits() -> None:
        # The slot (and the socket's next turn) waits for the engine call to
        # RETURN, not for this turn to end: the call cannot be preempted.
        try:
            await asyncio.gather(producer_task, return_exceptions=True)
        finally:
            with contextlib.suppress(Exception):
                await slot_cm.__aexit__(None, None, None)

    releaser = asyncio.create_task(_release_when_producer_exits())
    _releasers.add(releaser)
    releaser.add_done_callback(_releasers.discard)
    generation.running = releaser
    cancel_task = asyncio.create_task(cancel_event.wait())

    tokens = 0
    cancelled = False
    finished = False
    try:
        while True:
            getter = asyncio.create_task(queue.get())
            done, _pending = await asyncio.wait(
                {getter, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done:
                getter.cancel()
                cancelled = True
                break
            kind, payload = getter.result()
            if kind == "token":
                tokens += 1
                await _send(ws, {"type": "token", "delta": payload})
            elif kind == "error":
                await _send(ws, {"type": "error", "message": payload})
                finished = True
                break
            elif kind == "done":
                finished = True
                break
    finally:
        cancel_task.cancel()
        if not finished:
            # Cancelled, or the socket went away: ask the engine to stop so
            # the slot frees soon. ``producer_task`` is NOT cancelled — that
            # would only mark the future done while the thread runs on.
            signal.set()
            _cancel_engine(engine)

    if cancelled:
        # V6 ν2 — telemetry. The producer thread cannot be
        # interrupted without an engine-level cancellation API
        # (llama-cpp-python doesn't expose one), so when we cancel
        # mid-flight the dispatcher slot stays busy until the engine
        # returns by itself. We log a warning and bump
        # ``hfl_ws_cancel_orphans_total`` so capacity-planning knows
        # how often this happens.
        producer_finished = producer_task.done()
        try:
            from hfl.metrics import get_metrics

            get_metrics().record_ws_cancel(orphaned_slot=not producer_finished)
        except Exception:  # pragma: no cover — defensive
            pass
        if not producer_finished:
            logger.warning(
                "ws cancel left the dispatcher slot busy: the engine "
                "call for model %r is still running in the background",
                model_name,
            )
        await _send(ws, {"type": "cancelled", "tokens": tokens})
    else:
        await _send(ws, {"type": "done", "tokens": tokens})


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def _check_ws_auth_and_origin(ws: WebSocket) -> tuple[bool, str | None]:
    """Pre-accept gate for ``/ws/chat``.

    The HTTP ``APIKeyMiddleware`` and the ``CORSMiddleware`` only
    apply to standard HTTP routes — WebSocket upgrades bypass them.
    We mirror their two policies inline:

    1. **API key**: when ``state.api_key`` is configured, the
       handshake must carry it as either ``?api_key=<value>``,
       ``Authorization: Bearer <value>``, or ``X-API-Key: <value>``.
    2. **Origin**: when ``cors_origins`` / ``cors_allow_all`` is
       configured, the ``Origin`` header must be in the allow-list
       (or wildcard).

    Returns ``(ok, reason)``. ``reason`` carries the reject message
    that the close frame surfaces to the client.
    """
    import secrets as _secrets

    from hfl.api.state import get_state
    from hfl.config import config

    state = get_state()

    # Step 1: API key.
    #
    # SEC: headers are tried FIRST and the query string is the last resort.
    # A full URL ends up in reverse-proxy access logs, browser history and
    # ``Referer`` headers, so a key carried there is a key that leaks. The
    # query form cannot simply be removed — the browser WebSocket API
    # cannot set custom headers on the handshake — but nothing forces a
    # CLI or SDK client to use it, and until this change the query string
    # took priority even when a proper header was present.
    if state.api_key:
        provided: str | None = None
        auth = ws.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            provided = auth[7:]
        if not provided:
            provided = ws.headers.get("x-api-key")
        if not provided:
            provided = ws.query_params.get("api_key")
            if provided:
                logger.warning(
                    "WebSocket API key supplied in the query string; it will appear "
                    "in proxy logs and browser history. Prefer the Authorization or "
                    "X-API-Key header where the client allows it."
                )
        expected = state.api_key.encode()
        if not provided or not _secrets.compare_digest(provided.encode(), expected):
            return False, "unauthorized"

    # Step 2: Origin allow-list. Empty list + cors_allow_all=False
    # means same-origin only — and a missing Origin header is treated
    # as same-origin (the browser only sets it for cross-origin
    # requests).
    origin = ws.headers.get("origin")
    if origin:
        if config.cors_allow_all:
            return True, None
        if config.cors_origins and origin not in config.cors_origins:
            return False, f"origin not allowed: {origin}"
        if not config.cors_origins and not config.cors_allow_all:
            # Strict same-origin: reject any Origin we cannot match.
            return False, f"origin not allowed: {origin}"

    return True, None


@router.websocket("/ws/chat")
async def ws_chat(ws: WebSocket) -> None:
    """bidirectional chat with cancellation.

    The connection stays open across multiple chat turns; clients
    cancel a turn by sending ``{"type": "cancel"}`` mid-flight. A
    ``ping`` frame round-trips as ``pong`` so heartbeats can be
    layered without reconnecting.
    """
    ok, reason = _check_ws_auth_and_origin(ws)
    # Late import: server.py imports this module. A failed key counts — and
    # backs off — exactly as on HTTP, or the handshake was a free oracle for
    # guessing it at full speed.
    from hfl.api import server as _server
    from hfl.api.state import get_state

    if reason == "unauthorized":
        await _server._record_auth_failure(ws)
    elif ok and get_state().api_key:
        _server._clear_auth_failures(ws)
    if not ok:
        # Accept-then-close so the browser surfaces the reason
        # rather than a generic handshake failure.
        await ws.accept()
        await _send(ws, {"type": "error", "message": reason or "rejected"})
        await ws.close(code=1008)  # 1008 = "Policy Violation"
        return

    await ws.accept()

    drive_task: asyncio.Task[None] | None = None
    generation = _SocketGeneration()

    try:
        while True:
            try:
                raw = await ws.receive_text()
            except WebSocketDisconnect:
                return

            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                await _send(ws, {"type": "error", "message": "invalid JSON frame"})
                continue
            if not isinstance(frame, dict):
                await _send(ws, {"type": "error", "message": "frame must be a JSON object"})
                continue

            kind = frame.get("type")
            if kind == "chat":
                if drive_task is not None and not drive_task.done():
                    await _send(
                        ws,
                        {"type": "error", "message": "a chat turn is already in progress"},
                    )
                    continue
                # CON-2: run the turn in the background so this receive loop
                # keeps reading frames — notably ``cancel`` — while generation
                # is in flight. The cancel event is created here, before the
                # task is scheduled, so a ``cancel`` frame can never race ahead
                # of its creation nor be cleared out from under it.
                cancel_event = asyncio.Event()
                ws.scope["hfl_ws_cancel"] = cancel_event
                drive_task = asyncio.create_task(_drive_chat(ws, frame, cancel_event, generation))
                drive_task.add_done_callback(lambda t: _log_drive_result(ws, t))
            elif kind == "cancel":
                pending = ws.scope.get("hfl_ws_cancel")
                if pending is not None:
                    pending.set()
            elif kind == "ping":
                await _send(ws, {"type": "pong"})
            else:
                await _send(
                    ws,
                    {"type": "error", "message": f"unknown frame type: {kind!r}"},
                )
    finally:
        # Stop any in-flight generation before closing the socket.
        if drive_task is not None and not drive_task.done():
            drive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await drive_task
        try:
            await ws.close()
        except RuntimeError:
            pass
