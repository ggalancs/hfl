# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Section D: each ``HFL_*`` variable the code reads, set, and its effect seen.

Ids follow the audit's table (alphabetical). Each probe starts what the
variable changes and checks that it changed — not that ``hfl config`` echoes
it. The evidence says whether ``docs/env-vars.md`` documents it.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from pathlib import Path

import httpx
from local_audit import (
    QUESTION,
    Audit,
    Uncheckable,
    check,
    expect,
    need_apple_silicon,
    need_llama_server,
)

REPO = Path(__file__).resolve().parents[2]
DOCS = (REPO / "docs" / "env-vars.md").read_text()
USER = [{"role": "user", "content": QUESTION}]


def _doc(var: str) -> str:
    return "documented" if f"`{var}`" in DOCS else "NOT documented"


def _log(a: Audit) -> str:
    return max((a.work / "logs").glob("serve-*.log"), key=lambda p: p.stat().st_mtime).read_text(
        errors="replace"
    )


def _chat(base: str, model: str = "chat", **options: object) -> httpx.Response:
    body = {"model": model, "stream": False, "messages": USER, "options": options}
    return httpx.post(base + "/api/chat", json=body, timeout=600)


def _long(base: str, tokens: int) -> httpx.Response:
    """A generation that really runs ``tokens`` tokens: the short QUESTION ends in
    a fraction of a second whatever num_predict says, so no queue ever forms."""
    long = [
        {"role": "user", "content": "Count from 1 to 2000, one number per line, no commentary."}
    ]
    body = {"model": "chat", "stream": False, "messages": long, "options": {"num_predict": tokens}}
    return httpx.post(base + "/api/chat", json=body, timeout=600)


def probe(cid: str, var: str) -> object:
    def register(fn: object) -> object:
        def run(a: Audit) -> str:
            return f"{fn(a)} ({_doc(var)})"  # type: ignore[operator]

        return check(cid, var)(run)

    return register


@probe("D1", "HFL_ACCEPT_NETWORK_EXPOSURE")
def d1(a: Audit) -> str:
    with a.server("--host", "0.0.0.0", env={"HFL_ACCEPT_NETWORK_EXPOSURE": "true"}):
        pass
    return "0.0.0.0 unattended starts with it (refused without: A30)"


@probe("D2", "HFL_ALLOW_AGENT_LOOP")
def d2(a: Audit) -> str:
    body = {"model": "chat", "stream": False, "messages": USER, "agent_loop": True}
    with a.server() as base:
        off = httpx.post(base + "/api/chat", json=body, timeout=600).status_code
    with a.server(env={"HFL_ALLOW_AGENT_LOOP": "true"}) as base:
        on = httpx.post(base + "/api/chat", json=body, timeout=600).status_code
    expect(off == 403 and on == 200, (off, on))
    return "agent_loop 403 by default, 200 with it"


@probe("D3", "HFL_ALLOW_REMOTE_CODE")
def d3(a: Audit) -> str:
    code = "from hfl.security import remote_code_allowed as f; print(f())"
    import subprocess

    def value(env: dict) -> str:
        return subprocess.run(
            [a.python, "-c", code], capture_output=True, text=True, env={**a.env, **env}
        ).stdout.strip()

    off, on = value({}), value({"HFL_ALLOW_REMOTE_CODE": "1"})
    expect((off, on) == ("False", "True"), (off, on))
    return "trust_remote_code off by default, on with it (read by hfl.security)"


def _lan_ip() -> str:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("192.0.2.1", 9))  # no packet is sent
        return str(sock.getsockname()[0])


@probe("D4", "HFL_ALLOW_REMOTE_PULL")
def d4(a: Audit) -> str:
    """A client on the LAN address is a remote peer to HFL."""
    body = {"model": "nomic-ai/nomic-embed-text-v1.5-GGUF:Q2_K", "stream": False}
    key = {"Authorization": "Bearer k"}
    codes = []
    for extra in ({}, {"HFL_ALLOW_REMOTE_PULL": "true"}):
        with a.server("--host", "0.0.0.0", env={"HFL_API_KEY": "k", **extra}):
            remote = f"http://{_lan_ip()}:{a.port}"
            codes.append(
                httpx.post(remote + "/api/pull", json=body, headers=key, timeout=900).status_code
            )
    expect(codes == [403, 200], codes)
    return "a remote client: pull 403 by default, 200 with it"


@probe("D5", "HFL_API_KEY")
def d5(a: Audit) -> str:
    with a.server(env={"HFL_API_KEY": "env-key"}) as base:
        codes = (
            httpx.get(base + "/api/tags").status_code,
            httpx.get(base + "/api/tags", headers={"Authorization": "Bearer env-key"}).status_code,
        )
    expect(codes == (401, 200), codes)
    return "401 without the key, 200 with it"


def _admin_action(base: str) -> None:
    httpx.post(
        base + "/api/copy", json={"source": "chat", "destination": "d-audit-copy"}, timeout=60
    )
    httpx.request("DELETE", base + "/api/delete", json={"model": "d-audit-copy"}, timeout=60)


@probe("D6", "HFL_AUDIT_LOG_BACKUPS")
def d6(a: Audit) -> str:
    path = a.scratch / "audit-rot.log"
    with a.server(
        env={
            "HFL_AUDIT_LOG_PATH": str(path),
            "HFL_AUDIT_LOG_MAX_BYTES": "200",
            "HFL_AUDIT_LOG_BACKUPS": "2",
        }
    ) as base:
        for _ in range(4):
            _admin_action(base)
    backups = sorted(p.name for p in path.parent.glob("audit-rot.log.*"))
    expect(1 <= len(backups) <= 2, backups)
    return f"rotated, {len(backups)} backup(s) kept"


@probe("D7", "HFL_AUDIT_LOG_MAX_BYTES")
def d7(a: Audit) -> str:
    backups = list(a.scratch.glob("audit-rot.log.*"))
    expect(backups, "no rotation with a 200-byte cap (run with D6)")
    return "a 200-byte cap rotated the log"


@probe("D8", "HFL_AUDIT_LOG_PATH")
def d8(a: Audit) -> str:
    path = a.scratch / "audit.log"
    with a.server(env={"HFL_AUDIT_LOG_PATH": str(path)}) as base:
        _admin_action(base)
    expect(path.exists() and "copy" in path.read_text().lower(), "no audit entry")
    return "owner actions logged there"


@probe("D9", "HFL_DEBUG")
def d9(a: Audit) -> str:
    with a.server(env={"HFL_DEBUG": "1"}) as base:
        _chat(base, num_predict=2)
    expect("DEBUG" in _log(a), "no DEBUG lines")
    return "DEBUG lines in the log"


@probe("D10", "HFL_DEFAULT_CTX_SIZE")
def d10(a: Audit) -> str:
    with a.server(env={"HFL_DEFAULT_CTX_SIZE": "2048"}) as base:
        _chat(base, num_predict=2)
        ctx = httpx.get(base + "/api/ps").json()["models"][0]["details"].get("context_size")
    expect(ctx == 2048, ctx)
    return "the model loaded with a 2048 context"


@probe("D11", "HFL_DISABLE_MEMORY_PREFLIGHT")
def d11(a: Audit) -> str:
    with a.server(env={"HFL_MEMORY_BUDGET": "1", "HFL_DISABLE_MEMORY_PREFLIGHT": "1"}) as base:
        code = _chat(base, num_predict=2).status_code
    expect(code == 200, f"refused with the preflight off: {code}")
    return "a 1% budget does not stop a load with the preflight off"


@probe("D12", "HFL_DISABLE_MLX")
def d12(a: Audit) -> str:
    need_apple_silicon("MLX")
    with a.server() as base:  # the control: without it, MLX serves safetensors
        code = _chat(base, model="hfq", num_predict=2).status_code
    expect(
        code == 200 and "MLX model loaded" in _log(a), f"control: MLX did not serve hfq ({code})"
    )
    with a.server(env={"HFL_DISABLE_MLX": "1"}) as base:
        code = _chat(base, model="hfq", num_predict=2).status_code
    expect(code == 200, f"with MLX off the model did not answer: {code}")
    expect("MLX model loaded" not in _log(a), "loaded with MLX anyway")
    return "safetensors: MLX without it, another backend with it"


@probe("D13", "HFL_FLASH_ATTENTION")
def d13(a: Audit) -> str:
    with a.server(env={"HFL_FLASH_ATTENTION": "1", "HFL_DEBUG": "1"}) as base:
        _chat(base, num_predict=2)
    expect("flash" in _log(a).lower(), "no mention of flash attention")
    return "flash attention requested at load"


@probe("D14", "HFL_GENERATION_TIMEOUT")
def d14(a: Audit) -> str:
    with a.server(env={"HFL_GENERATION_TIMEOUT": "1"}) as base:
        _chat(base, num_predict=2)  # the load is not what the budget times
        out = _long(base, 4000)
    expect(
        out.status_code == 504,
        (
            out.status_code,
            out.json().get("eval_count") if out.status_code == 200 else out.text[:200],
        ),
    )
    return "a 1 s budget on a long generation: 504"


@probe("D15", "HFL_HOME")
def d15(a: Audit) -> str:
    expect((a.home / "models.json").exists(), "no registry under HFL_HOME")
    return "every check runs in its own HFL_HOME"


@probe("D16", "HFL_HOST")
def d16(a: Audit) -> str:
    out = a.cli("serve", "--port", "1", env={"HFL_HOST": "0.0.0.0"}, timeout=60)
    expect(out.returncode == 1 and "0.0.0.0" in out.stdout + out.stderr, out.stdout[-200:])
    return "serve takes its host from it (0.0.0.0, refused unattended)"


@probe("D17", "HFL_KEEP_ALIVE")
def d17(a: Audit) -> str:
    with a.server(env={"HFL_KEEP_ALIVE": "3s"}) as base:
        _chat(base, num_predict=2)
        time.sleep(25)  # 3 s keep-alive + the reaper's 15 s interval + slack
        left = httpx.get(base + "/api/ps").json()["models"]
    expect(not left, left)
    return "unloaded after 3 s idle (reaper runs every 15 s)"


@probe("D18", "HFL_KV_CACHE_TYPE")
def d18(a: Audit) -> str:
    import subprocess

    model = next(a.home.rglob("qwen2.5-0.5b-instruct-q4_k_m.gguf"), None)
    expect(model, "chat model not pulled")
    probe_ = (
        "from hfl.engine.llama_cpp import LlamaCppEngine\n"
        f"e = LlamaCppEngine(); e.load({str(model)!r}, n_ctx=512)\n"
        "print('TYPE_K', e._model.context_params.type_k)\n"
    )
    out = subprocess.run(
        [a.python, "-c", probe_],
        capture_output=True,
        text=True,
        timeout=300,
        env={**a.env, "HFL_KV_CACHE_TYPE": "q8_0"},
    )
    expect("TYPE_K 8" in out.stdout, f"KV cache not q8_0 (GGML type 8): {out.stdout[-200:]}")
    with a.server(env={"HFL_KV_CACHE_TYPE": "q8_0", "HFL_DEBUG": "1"}) as base:
        _chat(base, num_predict=2)
    expect(
        "KV cache quantised" in _log(a),
        "applied (llama context type_k=8, q8_0), but its log line never reaches the serve log: "
        "the load runs under _suppress_stderr, which also swallows HFL's own logging",
    )
    return "q8_0 KV cache applied and logged"


@probe("D19", "HFL_LANG")
def d19(a: Audit) -> str:
    out = a.ok("list", env={"HFL_LANG": "es"})
    expect("Modelos" in out or "Nombre" in out, out[:200])
    return "Spanish output"


@probe("D20", "HFL_LICENSE_POLICY")
def d20(a: Audit) -> str:
    body = {"model": "Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M", "stream": False}  # qwen-research
    with a.server() as base:
        refused = httpx.post(base + "/api/pull", json=body, timeout=300)
    expect(
        refused.status_code in (400, 403) and "licen" in refused.text.lower(), refused.text[:200]
    )
    return "a conditional license refused under the default policy (nothing accepted)"


@probe("D21", "HFL_LLAMA_SERVER_BIN")
def d21(a: Audit) -> str:
    out = a.cli(
        "serve",
        "--backend",
        "llama-server",
        "--port",
        "18779",
        env={"HFL_LLAMA_SERVER_BIN": "/nonexistent"},
        timeout=120,
    )
    text = out.stdout + out.stderr
    expect(out.returncode != 0 and "llama-server" in text and "Traceback" not in text, text[-300:])
    real = need_llama_server()
    with a.server("--backend", "llama-server", env={"HFL_LLAMA_SERVER_BIN": real}) as base:
        code = _chat(base, num_predict=2).status_code
    expect(code == 200 and "llama-server serving" in _log(a), f"with the real binary: {code}")
    return "missing binary: refused at start, clear message; real binary: serves"


@probe("D22", "HFL_LLM_LIBRARY")
def d22(a: Audit) -> str:
    need_llama_server()
    with a.server(env={"HFL_LLM_LIBRARY": "llama-server"}) as base:
        _chat(base, num_predict=2)
    expect("llama-server serving" in _log(a), "not served by llama-server")
    return "GGUF served by llama-server"


@probe("D23", "HFL_MAX_BLOB_BYTES")
def d23(a: Audit) -> str:
    import hashlib

    data = b"x" * 100
    with a.server(env={"HFL_MAX_BLOB_BYTES": "10"}) as base:
        out = httpx.post(
            f"{base}/api/blobs/sha256:{hashlib.sha256(data).hexdigest()}", content=data
        )
    expect(out.status_code == 413, f"cap enforced but answered {out.status_code}: {out.text[:120]}")
    return "a 100-byte blob over a 10-byte cap: 413"


@probe("D24", "HFL_MAX_LOADED_MODELS")
def d24(a: Audit) -> str:
    with a.server(env={"HFL_MAX_LOADED_MODELS": "1"}) as base:
        _chat(base, num_predict=2)
        _chat(base, model="think", num_predict=2)
        loaded = [m["name"] for m in httpx.get(base + "/api/ps").json()["models"]]
    expect(len(loaded) == 1, loaded)
    return "one model resident at a time"


@probe("D25", "HFL_MAX_QUEUE")
def d25(a: Audit) -> str:
    return _queue(a, {"HFL_MAX_QUEUE": "1"})


@probe("D26", "HFL_MAX_REQUEST_BYTES")
def d26(a: Audit) -> str:
    with a.server(env={"HFL_MAX_REQUEST_BYTES": "1000"}) as base:
        code = httpx.post(
            base + "/api/chat",
            json={"model": "chat", "messages": [{"role": "user", "content": "x" * 5000}]},
        ).status_code
    expect(code == 413, code)
    return "a 5 KB body over a 1 KB cap: 413"


@probe("D27", "HFL_MCP_AUTOLOAD")
def d27(a: Audit) -> str:
    config = a.scratch / "mcp.json"
    config.write_text(
        json.dumps({"servers": [{"id": "self", "target": f"stdio://{a.hfl} mcp serve"}]})
    )
    with a.server(env={"HFL_MCP_AUTOLOAD": str(config)}):
        log = _log(a)
    expect("mcp" in log.lower(), "nothing about MCP in the serve log; " + log[-200:])
    return "read at start"


@probe("D28", "HFL_MEMORY_BUDGET")
def d28(a: Audit) -> str:
    with a.server(env={"HFL_MEMORY_BUDGET": "1"}) as base:
        out = _chat(base, num_predict=2)
    expect(
        out.status_code in (503, 507) and "memory" in out.text.lower(),
        (out.status_code, out.text[:200]),
    )
    return f"a 1% budget refuses the load: {out.status_code}"


@probe("D29", "HFL_METRICS_PUBLIC")
def d29(a: Audit) -> str:
    with a.server(env={"HFL_API_KEY": "k"}) as base:
        closed = httpx.get(base + "/metrics").status_code
    with a.server(env={"HFL_API_KEY": "k", "HFL_METRICS_PUBLIC": "true"}) as base:
        opened = httpx.get(base + "/metrics").status_code
    expect((closed, opened) == (401, 200), (closed, opened))
    return "/metrics 401 behind the key, 200 with it public"


@probe("D30", "HFL_MLX_PROMPT_CACHE_BYTES")
def d30(a: Audit) -> str:
    need_apple_silicon("MLX")
    with a.server(env={"HFL_MLX_PROMPT_CACHE_BYTES": "0"}) as base:
        code = _chat(base, model="mlxq", num_predict=4).status_code
    expect(code == 200, code)
    return "0 runs MLX uncached and still answers"


@probe("D31", "HFL_MODEL_LOAD_TIMEOUT")
def d31(a: Audit) -> str:
    with a.server(env={"HFL_MODEL_LOAD_TIMEOUT": "0.001"}) as base:
        out = _chat(base, num_predict=2)
    expect(
        out.status_code >= 500 and "Traceback" not in out.text, (out.status_code, out.text[:200])
    )
    return f"a 1 ms load budget: {out.status_code}"


@probe("D32", "HFL_NO_MLX_HINT")
def d32(a: Audit) -> str:
    need_apple_silicon("the MLX hint")
    with a.server() as base:  # the control: without it, a GGUF load hints at MLX
        _chat(base, num_predict=2)
    expect("MLX build of the same model" in _log(a), "control: no hint without the variable")
    with a.server(env={"HFL_NO_MLX_HINT": "1"}) as base:
        _chat(base, num_predict=2)
    expect("MLX build of the same model" not in _log(a), "hint still logged")
    return "the MLX hint: logged without it, silenced with it"


@probe("D33", "HFL_NUM_PARALLEL")
def d33(a: Audit) -> str:
    need_llama_server()
    with a.server("--backend", "llama-server", env={"HFL_NUM_PARALLEL": "2"}) as base:
        _chat(base, num_predict=2)
    expect("shared by 2 parallel slots" in _log(a), "not 2 slots")
    return "llama-server with 2 slots"


@probe("D34", "HFL_ORIGINS")
def d34(a: Audit) -> str:
    origin = "https://ui.example"
    with a.server(env={"HFL_ORIGINS": origin}) as base:
        header = httpx.get(base + "/api/tags", headers={"Origin": origin}).headers.get(
            "access-control-allow-origin"
        )
    expect(header == origin, header)
    return "CORS allows the listed origin"


def _otel(a: Audit, var: str) -> str:
    import socket
    import subprocess

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    collector = a.scratch / "collector.py"
    collector.write_text(
        "import sys\nfrom http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "class H(BaseHTTPRequestHandler):\n"
        "    def log_message(self, *x): pass\n"
        "    def do_POST(self):\n"
        "        self.rfile.read(int(self.headers['Content-Length']))\n"
        "        print('SPAN', self.path, flush=True)\n"
        "        self.send_response(200); self.send_header('Content-Length','0')\n"
        "        self.end_headers()\n"
        f"HTTPServer(('127.0.0.1', {port}), H).serve_forever()\n"
    )
    out = a.scratch / "collector.out"
    with open(out, "w") as sink:
        proc = subprocess.Popen([a.python, "-u", str(collector)], stdout=sink)
    try:
        env = {
            "HFL_OTEL_ENABLED": "true",
            "HFL_OTEL_EXPORTER_ENDPOINT": f"http://127.0.0.1:{port}/v1/traces",
            "HFL_OTEL_SERVICE_NAME": "audit-svc",
        }
        with a.server(env=env) as base:
            _chat(base, num_predict=2)
            time.sleep(7)
        log = _log(a)
    finally:
        proc.kill()
    expect("SPAN" in out.read_text(), "no span received")
    if var == "HFL_OTEL_SERVICE_NAME":
        expect("audit-svc" in log, "service name not used")
    return "a span reached the local collector"


@probe("D35", "HFL_OTEL_ENABLED")
def d35(a: Audit) -> str:
    return _otel(a, "HFL_OTEL_ENABLED")


@probe("D36", "HFL_OTEL_EXPORTER_ENDPOINT")
def d36(a: Audit) -> str:
    return _otel(a, "HFL_OTEL_EXPORTER_ENDPOINT")


@probe("D37", "HFL_OTEL_SERVICE_NAME")
def d37(a: Audit) -> str:
    return _otel(a, "HFL_OTEL_SERVICE_NAME")


@probe("D38", "HFL_PORT")
def d38(a: Audit) -> str:
    import socket
    import subprocess

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen(
        [a.hfl, "serve"],
        env={**a.env, "HFL_PORT": str(port)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(120):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                    return f"serve listened on {port} from HFL_PORT"
            except httpx.HTTPError:
                time.sleep(0.5)
        expect(False, "not listening on HFL_PORT")
    finally:
        proc.terminate()
        proc.wait(timeout=60)
    return ""


def _queue(a: Audit, env: dict) -> str:
    env = {"HFL_QUEUE_MAX_INFLIGHT": "1", "HFL_QUEUE_ACQUIRE_TIMEOUT": "60", **env}
    with a.server(env=env) as base:
        _chat(base, num_predict=2)
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            codes = list(pool.map(lambda _: _long(base, 300).status_code, range(4)))
    expect(429 in codes, codes)
    return f"a full queue answers 429: {codes}"


@probe("D39", "HFL_QUEUE_ACQUIRE_TIMEOUT")
def d39(a: Audit) -> str:
    with a.server(env={"HFL_QUEUE_ACQUIRE_TIMEOUT": "0.1", "HFL_QUEUE_MAX_SIZE": "8"}) as base:
        _chat(base, num_predict=2)
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            codes = list(pool.map(lambda _: _long(base, 300).status_code, range(3)))
    expect(503 in codes, codes)
    return f"a 0.1 s wait: 503 for the queued: {codes}"


@probe("D40", "HFL_QUEUE_MAX_INFLIGHT")
def d40(a: Audit) -> str:
    return _queue(a, {"HFL_QUEUE_MAX_SIZE": "1"})


@probe("D41", "HFL_QUEUE_MAX_SIZE")
def d41(a: Audit) -> str:
    return _queue(a, {"HFL_QUEUE_MAX_SIZE": "1"})


def _rate(a: Audit, env: dict) -> list[int]:
    with a.server(env=env) as base:
        return [httpx.get(base + "/api/tags").status_code for _ in range(4)]


@probe("D42", "HFL_RATE_LIMIT_ENABLED")
def d42(a: Audit) -> str:
    off = _rate(a, {"HFL_RATE_LIMIT_ENABLED": "false", "HFL_RATE_LIMIT_REQUESTS": "2"})
    expect(429 not in off, off)
    return "off: no 429 over the limit"


@probe("D43", "HFL_RATE_LIMIT_REQUESTS")
def d43(a: Audit) -> str:
    codes = _rate(a, {"HFL_RATE_LIMIT_REQUESTS": "2", "HFL_RATE_LIMIT_WINDOW": "60"})
    expect(429 in codes, codes)
    return f"2 per window: {codes}"


@probe("D44", "HFL_RATE_LIMIT_WINDOW")
def d44(a: Audit) -> str:
    with a.server(env={"HFL_RATE_LIMIT_REQUESTS": "1", "HFL_RATE_LIMIT_WINDOW": "2"}) as base:
        first = [httpx.get(base + "/api/tags").status_code for _ in range(2)]
        time.sleep(3)
        after = httpx.get(base + "/api/tags").status_code
    expect(first[1] == 429 and after == 200, (first, after))
    return "a 2 s window: 429, then 200 once it passed"


@probe("D45", "HFL_SANDBOX")
def d45(a: Audit) -> str:
    with a.server(env={"HFL_SANDBOX": "macos"}) as base:
        code = _chat(base, num_predict=2).status_code
    log = _log(a)
    expect(code == 200 and "sandbox" in log.lower(), (code, log[-200:]))
    return "serves under the macOS sandbox"


@probe("D46", "HFL_STREAM_QUEUE_GET_TIMEOUT")
def d46(a: Audit) -> str:
    raise Uncheckable("internal stream timeout: needs a stalled engine to observe")


@probe("D47", "HFL_STREAM_QUEUE_PUT_TIMEOUT")
def d47(a: Audit) -> str:
    raise Uncheckable("internal stream timeout: needs a stalled client to observe")


@probe("D48", "HFL_TOOLS")
def d48(a: Audit) -> str:
    import subprocess

    request = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "audit", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    out = subprocess.run(
        [a.hfl, "mcp", "serve"],
        input="\n".join(json.dumps(r) for r in request) + "\n",
        capture_output=True,
        text=True,
        timeout=60,
        env={**a.env, "HFL_TOOLS": "web_fetch"},
    )
    expect(
        "web_fetch" in out.stdout and "web_search" not in out.stdout,
        (out.stdout + out.stderr)[-300:],
    )
    return "only the listed tool exposed"


@probe("D49", "HFL_VLLM_ERROR_PUT_TIMEOUT")
def d49(a: Audit) -> str:
    raise Uncheckable("vLLM only (Linux + CUDA)")


@probe("D50", "HFL_VLLM_SHUTDOWN_JOIN_TIMEOUT")
def d50(a: Audit) -> str:
    raise Uncheckable("vLLM only (Linux + CUDA)")


@probe("D51", "HFL_VRAM_OVERRIDE_GIB")
def d51(a: Audit) -> str:
    out = a.ok("doctor", env={"HFL_VRAM_OVERRIDE_GIB": "3"})
    expect("3" in out and ("GiB" in out or "GB" in out), out[-400:])
    return "doctor reports the overridden 3 GiB"


@probe("D52", "HFL_WEB_SEARCH_BACKEND")
def d52(a: Audit) -> str:
    with a.server(env={"HFL_WEB_SEARCH_BACKEND": "brave"}) as base:
        out = httpx.post(base + "/api/web_search", json={"query": "x"}, timeout=60)
    expect(
        out.status_code >= 400 and "brave" in out.text.lower(), (out.status_code, out.text[:200])
    )
    return "brave without its key: a clear error naming it"


@check("D53", "documented OLLAMA_* fallbacks are read")
def d53(a: Audit) -> str:
    import re

    documented = sorted(set(re.findall(r"`(OLLAMA_[A-Z_]+)`", DOCS)))
    expect(documented, "no OLLAMA_* variable found in docs/env-vars.md")
    source = "\n".join(p.read_text() for p in (REPO / "src" / "hfl").rglob("*.py"))
    unread = [v for v in documented if f'"{v}"' not in source]
    expect(not unread, f"documented but never read: {unread}")
    return f"{len(documented)} documented, all read by the code"
