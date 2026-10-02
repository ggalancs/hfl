# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A client that stops reading must not hold the server's resources.

Measured (audit G4): a client asked for a long stream, read nothing more
and kept the connection open. Every send blocked, so the request's model
lease and queue slot were held for as long as it liked — four such clients
took a model's four slots. ``ModelLeaseMiddleware`` now gives each send a
deadline (``HFL_STREAM_QUEUE_PUT_TIMEOUT``) and drops the response after it.
This runs a real uvicorn server and a real socket that stops reading.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest


def _serve(app):
    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    return server, thread, port


@pytest.mark.slow
def test_a_client_that_stops_reading_is_dropped(monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route

    from hfl.api.server import ModelLeaseMiddleware
    from hfl.api.streaming import ClosingStreamingResponse
    from hfl.config import config

    monkeypatch.setattr(config, "stream_queue_put_timeout", 2.0)
    ended = threading.Event()

    async def endless():
        try:
            while True:
                yield b"x" * 65536  # far more than any socket buffer holds
        finally:
            ended.set()  # the body was given up: what it held is released

    async def route(request):
        return ClosingStreamingResponse(endless())

    app = ModelLeaseMiddleware(Starlette(routes=[Route("/s", route)]))
    server, thread, port = _serve(app)
    stalled = socket.socket()
    stalled.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    stalled.connect(("127.0.0.1", port))
    try:
        stalled.sendall(b"GET /s HTTP/1.1\r\nHost: x\r\n\r\n")
        assert stalled.recv(64).startswith(b"HTTP/1.1 200")  # it started, then reads nothing
        # Within the deadline and a margin; without the close it waited for
        # the garbage collector (measured: only gc.collect() released it).
        assert ended.wait(timeout=10), "the response kept waiting on a client that reads nothing"
    finally:
        stalled.close()
        server.should_exit = True
        thread.join(timeout=10)
