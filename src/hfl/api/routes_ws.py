# SPDX-License-Identifier: HRUL-1.0
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
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

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


async def _drive_chat(ws: WebSocket, frame: dict[str, Any]) -> None:
    """Run one chat turn over the WebSocket.

    Cancellation contract: a ``cancel`` frame from the client
    interrupts the generation by setting an internal asyncio.Event;
    the driver checks it on every produced token and exits cleanly.
    """
    try:
        model_name, messages = _validate_chat_frame(frame)
    except ValueError as exc:
        await _send(ws, {"type": "error", "message": str(exc)})
        return

    from hfl.api.model_loader import load_llm
    from hfl.engine.base import ChatMessage, GenerationConfig

    try:
        engine, _ = await load_llm(model_name)
    except Exception as exc:
        await _send(ws, {"type": "error", "message": f"load_llm failed: {exc}"})
        return

    if engine is None:
        await _send(ws, {"type": "error", "message": "engine not available"})
        return

    chat_msgs = [
        ChatMessage(role=str(m.get("role")), content=str(m.get("content") or "")) for m in messages
    ]
    options = frame.get("options") or {}
    cfg = GenerationConfig(
        max_tokens=int(options.get("max_tokens", 0) or 0),
        temperature=float(options.get("temperature", 0.7) or 0.7),
        top_p=float(options.get("top_p", 0.9) or 0.9),
    )

    await _send(ws, {"type": "ready", "model": model_name})

    cancel_event: asyncio.Event = ws.scope.setdefault("hfl_ws_cancel", asyncio.Event())
    cancel_event.clear()

    # Run the sync chat_stream in a thread so the event loop can
    # service inbound ``cancel`` frames concurrently.
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def _producer() -> None:
        try:
            for token in engine.chat_stream(chat_msgs, cfg):
                queue.put_nowait(("token", token))
        except Exception as exc:  # pragma: no cover — surface as error frame
            queue.put_nowait(("error", str(exc)))
        finally:
            queue.put_nowait(("done", None))

    producer_task = asyncio.create_task(asyncio.to_thread(_producer))
    cancel_task = asyncio.create_task(cancel_event.wait())

    tokens = 0
    cancelled = False
    try:
        while True:
            getter = asyncio.create_task(queue.get())
            done, pending = await asyncio.wait(
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
                break
            elif kind == "done":
                break
    finally:
        cancel_task.cancel()
        # The thread keeps running until the engine finishes;
        # we can't preempt it, but the stream consumer is gone.
        producer_task.cancel()

    if cancelled:
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
    if state.api_key:
        provided: str | None = ws.query_params.get("api_key")
        if not provided:
            auth = ws.headers.get("authorization", "")
            if auth.startswith("Bearer "):
                provided = auth[7:]
        if not provided:
            provided = ws.headers.get("x-api-key")
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
    """V4: bidirectional chat with cancellation.

    The connection stays open across multiple chat turns; clients
    cancel a turn by sending ``{"type": "cancel"}`` mid-flight. A
    ``ping`` frame round-trips as ``pong`` so heartbeats can be
    layered without reconnecting.
    """
    ok, reason = _check_ws_auth_and_origin(ws)
    if not ok:
        # Accept-then-close so the browser surfaces the reason
        # rather than a generic handshake failure.
        await ws.accept()
        await _send(ws, {"type": "error", "message": reason or "rejected"})
        await ws.close(code=1008)  # 1008 = "Policy Violation"
        return

    await ws.accept()

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
                await _drive_chat(ws, frame)
            elif kind == "cancel":
                cancel_event = ws.scope.get("hfl_ws_cancel")
                if cancel_event is not None:
                    cancel_event.set()
            elif kind == "ping":
                await _send(ws, {"type": "pong"})
            else:
                await _send(
                    ws,
                    {"type": "error", "message": f"unknown frame type: {kind!r}"},
                )
    finally:
        try:
            await ws.close()
        except RuntimeError:
            pass
