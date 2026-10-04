"""G. Failures: what breaks, and how HFL comes back.

Production plan 1.3. Each check breaks something for real — kills a
process, fills a disk, cuts the network, stalls a client, stops the server
with requests waiting — and passes only when nothing hangs, nothing is left
running, the client is told, and HFL serves again afterwards. Every wait
has its own deadline: a check never depends on the thing it breaks to end.
"""

from __future__ import annotations

import concurrent.futures
import json
import select
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time

import httpx
from local_audit import (
    QUESTION,
    WINDOWS,
    Audit,
    Parts,
    Uncheckable,
    check,
    expect,
    need_llama_server,
    registered_aliases,
)

USER = [{"role": "user", "content": QUESTION}]
# A reply that is really long: a counting sequence to continue, without the
# chat template, runs to num_predict (measured: 2000 of 2000 tokens). A chat
# request to "count to 3000" got a short answer, and the checks then tested
# a reply that had already ended.
COUNT = ", ".join(str(i) for i in range(1, 41)) + ","


def _long(tokens: int, stream: bool) -> dict:
    return {
        "model": "chat", "prompt": COUNT, "raw": True, "stream": stream,
        "options": {"num_predict": tokens, "temperature": 0},
    }  # fmt: skip


def _psutil():
    try:
        import psutil
    except ImportError as exc:
        raise Uncheckable("needs psutil in the harness venv") from exc
    return psutil


def _llama_servers(a: Audit) -> list:
    """The llama-server processes serving this audit's models — found on the
    system, not asked of HFL (a check must not trust what it checks)."""
    psutil = _psutil()
    found = []
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            name = proc.info["name"] or ""
            cmdline = proc.info["cmdline"] or []
            if name.startswith("llama-server") and any(str(a.home) in c for c in cmdline):
                found.append(proc)
        except psutil.Error:
            pass
    return found


def _left_behind(a: Audit) -> list:
    """The llama-servers still running once their server is gone, after the
    guard's own bound: it notices its parent's death within a second and
    gives llama-server ten to stop (on Windows the harness stops HFL with
    TerminateProcess, so the guard is what cleans up)."""
    deadline = time.monotonic() + 20
    found = _llama_servers(a)
    while found and time.monotonic() < deadline:
        time.sleep(0.5)
        found = _llama_servers(a)
    return found


def _answers(base: str, model: str = "chat", timeout: float = 300) -> str:
    """A short chat answer, or what went wrong."""
    body = {"model": model, "stream": False, "messages": USER, "options": {"num_predict": 8}}
    try:
        r = httpx.post(base + "/api/chat", json=body, timeout=timeout)
    except httpx.HTTPError as exc:
        return f"{type(exc).__name__}: {exc}"
    if r.status_code != 200:
        return f"{r.status_code}: {r.text[:200]}"
    return r.json().get("message", {}).get("content", "")


# -- G1 ------------------------------------------------------------------------


@check("G1", "llama-server killed in the middle of a reply", needs=("A24",))
def g1(a: Audit) -> str:
    """The child serving a streamed reply dies (an OOM kill, a crash). The
    stream must end — with an error, not as a finished reply — and the next
    request must load the model again and answer."""
    need_llama_server()
    part = Parts()
    with a.server() as base:
        expect(_answers(base).strip(), "chat did not answer before the kill")
        body = _long(3000, stream=True)
        lines: list[str] = []
        killed: list = []
        ended = "closed"
        started = time.monotonic()
        try:
            with httpx.stream(
                "POST", base + "/api/generate", json=body, timeout=httpx.Timeout(120, connect=10)
            ) as response:
                for line in response.iter_lines():
                    lines.append(line)
                    if len(lines) == 10:
                        killed = _llama_servers(a)
                        for proc in killed:
                            proc.kill()
        except httpx.HTTPError as exc:
            ended = f"{type(exc).__name__}"
        took = time.monotonic() - started
        part("a llama-server was serving it", lambda: expect(killed, "none found"))
        part("the stream ends (no hang)", lambda: expect(took < 110, f"{took:.0f}s"))
        last: dict = {}
        if lines:
            try:
                last = json.loads(lines[-1])
            except ValueError:
                last = {}
        finished_quietly = ended == "closed" and last.get("done") and not last.get("error")
        part(
            "the client is told it broke (not a finished reply)",
            lambda: expect(not finished_quietly, f"last line: {lines[-1][:200] if lines else ''}"),
        )
        part("HFL itself still runs", lambda: expect(a.proc and a.proc.poll() is None, "exited"))
        again = _answers(base)
        part("the next request answers", lambda: expect("paris" in again.lower(), again[:200]))
        stale = [p for p in killed if p.is_running() and p.status() != "zombie"]
        part("the killed one is gone", lambda: expect(not stale, [p.pid for p in stale]))
    leftovers = _left_behind(a)
    part("nothing left after the server stops", lambda: expect(not leftovers, len(leftovers)))
    return part.verdict()


# -- G2 ------------------------------------------------------------------------


@check("G2", "disk full in the middle of a download")
def g2(a: Audit) -> str:
    """A 900 MB volume, a 530 MB model, and the volume filled once the
    download is under way: the pull must stop with a message (no traceback),
    register nothing, and complete once there is room again."""
    if sys.platform != "darwin":
        raise Uncheckable("needs a small volume of its own (hdiutil, macOS)")
    part = Parts()
    image = a.work / "g2.dmg"
    volume = a.work / "g2-volume"
    subprocess.run(["hdiutil", "detach", str(volume), "-force"], capture_output=True, timeout=60)
    if image.exists():
        image.unlink()  # a disk image this check created on its last run
    made = subprocess.run(
        ["hdiutil", "create", "-size", "900m", "-fs", "APFS", "-volname", "hflg2", str(image)],
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    expect(made.returncode == 0, made.stderr[-300:])
    volume.mkdir(exist_ok=True)
    attached = subprocess.run(
        ["hdiutil", "attach", "-nobrowse", "-mountpoint", str(volume), str(image)],
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    expect(attached.returncode == 0, attached.stderr[-300:])
    try:
        home = volume / "home"
        home.mkdir()
        env = {**a.env, "HFL_HOME": str(home), "HF_HOME": str(volume / "hf")}
        pull = [a.hfl, "pull", "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "-q", "Q8_0", "--alias", "g2",
                "--skip-license"]  # fmt: skip
        free_at_start = shutil.disk_usage(volume).free
        log_path = a.work / "logs" / "g2-pull.log"
        log = open(log_path, "wb")  # a pipe nobody reads would block the pull
        proc = subprocess.Popen(
            pull, env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, cwd=a.scratch,
        )  # fmt: skip
        filler = volume / "filler"
        filled = False
        deadline = time.monotonic() + 600
        while proc.poll() is None and time.monotonic() < deadline:
            if not filled and free_at_start - shutil.disk_usage(volume).free > 50 * 2**20:
                with open(filler, "wb") as out:
                    chunk = b"\0" * 2**20
                    try:
                        while shutil.disk_usage(volume).free > 8 * 2**20:
                            out.write(chunk)
                            out.flush()
                    except OSError:
                        pass  # full: what this check wants
                filled = True
            time.sleep(0.1)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)
            expect(False, "the pull did not stop within 10 minutes of a full disk")
        log.close()
        said = log_path.read_text(errors="replace")
        expect(filled, f"the download finished before the disk could be filled: {said[-200:]}")
        part("the pull stops with an error", lambda: expect(proc.returncode != 0, said[-300:]))
        part("no traceback", lambda: expect("Traceback" not in said, said[-500:]))
        part(
            "it says the disk is full",
            lambda: expect(any(w in said.lower() for w in ("space", "disk", "full")), said[-300:]),
        )
        part("nothing registered", lambda: expect("g2" not in registered_aliases(home), "g2"))
        filler.unlink()
        again = subprocess.run(
            pull, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL,
            timeout=1200, cwd=a.scratch,
        )  # fmt: skip
        part(
            "with room again, the same pull completes",
            lambda: expect(
                again.returncode == 0 and "g2" in registered_aliases(home),
                (again.stdout + again.stderr)[-300:],
            ),
        )
    finally:
        subprocess.run(
            ["hdiutil", "detach", str(volume), "-force"], capture_output=True, timeout=60
        )
    return part.verdict()


# -- G3 ------------------------------------------------------------------------


class _Relay(socketserver.ThreadingTCPServer):
    """An HTTPS proxy (CONNECT) that can drop every connection at once: the
    Hub going away in the middle of a download, as the client sees it."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _RelayHandler)
        self.relayed = 0
        self.cut = threading.Event()
        self.sockets: list[socket.socket] = []
        self.lock = threading.Lock()

    def drop(self) -> None:
        self.cut.set()
        with self.lock:
            for sock in self.sockets:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()
        self.shutdown()
        self.server_close()


class _RelayHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        relay: _Relay = self.server  # type: ignore[assignment]
        head = b""
        while b"\r\n\r\n" not in head:
            data = self.request.recv(4096)
            if not data:
                return
            head += data
        target = head.split(b" ", 2)[1].decode()
        host, _, port = target.rpartition(":")
        try:
            upstream = socket.create_connection((host, int(port)), timeout=30)
        except OSError:
            self.request.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            return
        self.request.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        with relay.lock:
            relay.sockets += [self.request, upstream]
        pair = {self.request: upstream, upstream: self.request}
        while not relay.cut.is_set():
            try:
                ready, _, _ = select.select(list(pair), [], [], 0.5)
                for sock in ready:
                    data = sock.recv(65536)
                    if not data:
                        return
                    pair[sock].sendall(data)
                    if sock is upstream:
                        relay.relayed += len(data)
            except OSError:
                return


@check("G3", "the Hub drops in the middle of a download")
def g3(a: Audit) -> str:
    """Every connection to the Hub cut at once after 40 MB of a download
    (through a local proxy that then stops answering): the pull must stop in
    bounded time with a message, register nothing, and complete when the
    network is back."""
    part = Parts()
    home = a.work / "g3-home"
    home.mkdir(exist_ok=True)
    base_env = {**a.env, "HFL_HOME": str(home), "HF_HOME": str(a.work / "g3-hf")}
    if "g3" in registered_aliases(home):
        # The model this check pulled last time: removed by HFL itself, so
        # the download starts from nothing again.
        gone = a.cli("rm", "g3", "--yes", env=base_env)
        expect(gone.returncode == 0, gone.stdout[-300:])
    relay = _Relay()
    threading.Thread(target=relay.serve_forever, daemon=True).start()
    proxy = f"http://127.0.0.1:{relay.server_address[1]}"
    env = {
        **a.env, "HFL_HOME": str(home), "HF_HOME": str(a.work / "g3-hf"),
        "HTTPS_PROXY": proxy, "HTTP_PROXY": proxy, "https_proxy": proxy, "http_proxy": proxy,
        "NO_PROXY": "", "no_proxy": "", "HF_HUB_DISABLE_XET": "1",
    }  # fmt: skip
    pull = [a.hfl, "pull", "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "-q", "Q4_K_M", "--alias", "g3",
            "--skip-license"]  # fmt: skip
    log_path = a.work / "logs" / "g3-pull.log"
    log = open(log_path, "wb")  # a pipe nobody reads would block the pull
    proc = subprocess.Popen(
        pull, env=env, stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, cwd=a.scratch,
    )  # fmt: skip
    deadline = time.monotonic() + 600
    while proc.poll() is None and relay.relayed < 40 * 2**20 and time.monotonic() < deadline:
        time.sleep(0.05)
    relayed = relay.relayed
    relay.drop()
    cut_at = time.monotonic()
    try:
        proc.wait(timeout=300)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)
    log.close()
    said = log_path.read_text(errors="replace")
    expect(
        relayed >= 40 * 2**20,
        f"the download went around the proxy ({relayed} bytes): {said[-200:]}",
    )
    took = time.monotonic() - cut_at
    part("the pull stops (no hang)", lambda: expect(proc.returncode is not None and took < 290,
                                                     f"{took:.0f}s"))  # fmt: skip
    part("with an error", lambda: expect(proc.returncode not in (0, None), said[-300:]))
    part("no traceback", lambda: expect("Traceback" not in said, said[-500:]))
    part("nothing registered", lambda: expect("g3" not in registered_aliases(home), "g3"))
    clean = {k: v for k, v in env.items() if k.lower() not in ("https_proxy", "http_proxy")}
    again = subprocess.run(
        pull, env=clean, capture_output=True, text=True, stdin=subprocess.DEVNULL,
        timeout=1200, cwd=a.scratch,
    )  # fmt: skip
    part(
        "with the network back, the same pull completes",
        lambda: expect(
            again.returncode == 0 and "g3" in registered_aliases(home),
            (again.stdout + again.stderr)[-300:],
        ),
    )
    return f"{part.verdict()}; cut after {relayed // 2**20} MB, stopped {took:.0f}s later"


# -- G4 ------------------------------------------------------------------------


def _send_queue(server_port: int, client_port: int) -> int | None:
    """Bytes the server has written to a connection and its client has not
    read (Linux: the ``tx_queue`` of /proc/net/tcp), or None when gone."""
    want = (f":{server_port:04X}", f":{client_port:04X}")
    for line in open("/proc/net/tcp").read().splitlines()[1:]:
        fields = line.split()
        if fields[1].endswith(want[0]) and fields[2].endswith(want[1]):
            return int(fields[4].split(":")[0], 16)
    return None


def _wait_until_stalled(base: str, server_port: int, client_port: int) -> None:
    """Until the stalled connection's buffer is full, so the server's writes
    block. Linux grows a connection's send buffer while its client reads
    nothing, up to ``tcp_wmem``'s maximum (4 MiB by default): on an L4 a
    20,000-token reply never filled it, the stream was simply busy, and the
    other model's 60-second wait ran out (Modal, 2026-10-04). With the buffer
    capped at 256 KB the same server let the model go 15 s after it filled.
    """
    cap = int(open("/proc/sys/net/ipv4/tcp_wmem").read().split()[2])
    last, steady, deadline = -1, 0, time.monotonic() + 240
    while time.monotonic() < deadline:
        queued = _send_queue(server_port, client_port)
        if (
            queued is None
            or httpx.get(base + "/healthz", timeout=10).json().get("queue_in_flight", 0) == 0
        ):
            raise Uncheckable(
                f"the reply ended before it filled the connection's buffer "
                f"(Linux grows it up to tcp_wmem's {cap / 2**20:.1f} MiB); "
                "tests/test_stalled_client.py covers the mechanism"
            )
        steady = steady + 1 if queued == last and queued > 0 else 0
        if steady >= 2:  # unchanged for 10 s: the writes block
            return
        last = queued
        time.sleep(5)
    raise Uncheckable(
        f"the connection's buffer was still growing after 240 s ({last / 2**20:.1f} MiB "
        f"of tcp_wmem's {cap / 2**20:.1f} MiB): this model does not fill it in time"
    )


@check("G4", "a client that stops reading its stream", needs=("A24",))
def g4(a: Audit) -> str:
    """A client asks for a long streamed reply, then reads nothing and keeps
    the connection open. The model must not stay held by it: with one model
    at a time, another model must load within the stream's put timeout plus
    a margin, while that client still hangs on."""
    part = Parts()
    env = {"HFL_MAX_LOADED_MODELS": "1", "HFL_STREAM_QUEUE_PUT_TIMEOUT": "15"}
    with a.server(env=env) as base:
        expect(_answers(base).strip(), "chat did not answer")
        # The stall shows only once the reply outgrows the network buffers;
        # on a slow generation (a CPU) that takes minutes, and the request is
        # then simply busy, which is right. tests/test_stalled_client.py
        # covers the mechanism there.
        timed = httpx.post(base + "/api/generate", json=_long(300, stream=False), timeout=300)
        reply = timed.json()
        rate = reply.get("eval_count", 0) / max(reply.get("eval_duration", 0) / 1e9, 1e-9)
        # On Linux the check watches the buffer fill itself (below), at any speed.
        if rate < 150 and not sys.platform.startswith("linux"):
            raise Uncheckable(
                f"generation at {rate:.0f} tok/s: too slow to fill the buffers in time "
                "(tests/test_stalled_client.py covers the mechanism)"
            )
        port = int(base.rsplit(":", 1)[1])
        long = _long(20000, stream=True)
        long["options"]["num_ctx"] = 32768
        body = json.dumps(long).encode()
        stalled = socket.create_connection(("127.0.0.1", port), timeout=10)
        stalled.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        stalled.sendall(
            b"POST /api/generate HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        head = stalled.recv(512)  # the headers and a first token: it has started
        expect(b" 200 " in head.split(b"\r\n", 1)[0], f"the stalled request: {head[:120]!r}")
        time.sleep(5)
        health = httpx.get(base + "/healthz", timeout=10).json()
        expect(
            health.get("queue_in_flight", 0) >= 1,
            f"the stalled request is not in flight any more: {health}",
        )
        if sys.platform.startswith("linux"):
            _wait_until_stalled(base, port, stalled.getsockname()[1])
        started = time.monotonic()
        # Another model, with one model at a time: chat has to go. A
        # request that holds chat would keep it (and this one waiting).
        other = _answers(base, model="stories", timeout=180)
        took = time.monotonic() - started
        part(
            "another model loads while the client stalls (within 120 s)",
            lambda: expect(took < 120 and other.strip() and not other[:3].isdigit(),
                           f"{took:.0f}s: {other[:200]}"),
        )  # fmt: skip
        part("HFL still runs", lambda: expect(a.proc and a.proc.poll() is None, "exited"))
        stalled.close()
        again = _answers(base)
        part(
            "after the client goes, chat answers",
            lambda: expect("paris" in again.lower(), again[:200]),
        )
    return part.verdict() + f"; the other model answered {took:.0f}s after the stall"


# -- G5 ------------------------------------------------------------------------


@check("G5", "the server stopped with requests waiting", needs=("A24",))
def g5(a: Audit) -> str:
    """One request at a time, four sent at once, and the server stopped
    (SIGTERM, as systemd or Docker stop it) while three wait: every client
    must get an answer or a closed connection — none left hanging — the
    server must exit, its llama-server with it, and a new one must serve."""
    if WINDOWS:
        raise Uncheckable("SIGTERM: POSIX")
    part = Parts()
    env = {"HFL_NUM_PARALLEL": "1"}
    outcomes: list[str] = []
    with a.server(env=env) as base:
        expect(_answers(base).strip(), "chat did not answer")
        proc = a.proc
        assert proc is not None
        body = _long(1500, stream=False)

        def ask(_: int) -> str:
            try:
                r = httpx.post(
                    base + "/api/chat", json=body, timeout=httpx.Timeout(180, connect=10)
                )
                return str(r.status_code)
            except httpx.HTTPError as exc:
                return type(exc).__name__

        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(ask, i) for i in range(4)]
            time.sleep(3)
            proc.send_signal(signal.SIGTERM)
            stopped = time.monotonic()
            for future in futures:
                try:
                    outcomes.append(future.result(timeout=200))
                except concurrent.futures.TimeoutError:
                    outcomes.append("HUNG")
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            pass
        exit_after = time.monotonic() - stopped
        part("no client left hanging", lambda: expect("HUNG" not in outcomes, outcomes))
        part("the server exits", lambda: expect(proc.poll() is not None, f"{exit_after:.0f}s"))
    leftovers = _left_behind(a)
    part("no llama-server left behind", lambda: expect(not leftovers, [p.pid for p in leftovers]))
    with a.server() as base:
        again = _answers(base)
        part("a new server serves", lambda: expect("paris" in again.lower(), again[:200]))
    return f"{part.verdict()}; clients got {outcomes}"
