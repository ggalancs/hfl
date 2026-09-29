# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Section E: each engine, through the same HTTP checks; and the routes that
need a speech, transcription or image model (section B ids)."""

from __future__ import annotations

import concurrent.futures
import io
import json
import math
import re
import shutil
import subprocess
import time
import wave
from pathlib import Path

import httpx
from local_audit import (
    QUESTION,
    Audit,
    Parts,
    Uncheckable,
    check,
    expect,
    need_apple_silicon,
    need_llama_server,
)

from audit_checks.c_extras import EXTRA_ID

REPO = Path(__file__).resolve().parents[2]
USER = [{"role": "user", "content": QUESTION}]
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _paris(text: str | None) -> bool:
    return "paris" in (text or "").lower()


def suite(
    part: Parts, base: str, model: str, *, logprobs: bool = True, formats: bool = True
) -> None:
    """The checks every chat engine must pass, over the three APIs.
    ``formats``: whether the engine constrains output to a format; one that
    cannot must refuse a format, never answer prose instead."""
    c = httpx.Client(base_url=base, timeout=600)

    def format_json() -> None:
        prose = "Name the capital of France."
        for path, body, text in (
            ("/api/generate", {"prompt": prose, "format": "json"}, lambda r: r["response"]),
            (
                "/api/chat",
                {"messages": [{"role": "user", "content": prose}], "format": "json"},
                lambda r: r["message"]["content"],
            ),
        ):
            # Greedy: sampled, a 0.5B model's "json" sometimes spends the whole
            # budget inside one string (15-16 of 20 valid either way, measured) —
            # a check that passed or failed by luck.
            options = {"num_predict": 200, "temperature": 0}
            payload = {"model": model, "stream": False, "options": options, **body}
            out = c.post(path, json=payload)
            if not formats:
                expect(out.status_code == 400 and "constrain" in out.text, (path, out.status_code))
                continue
            expect(out.status_code == 200, (path, out.status_code, out.text[:120]))
            try:
                json.loads(text(out.json()))
            except ValueError:
                expect(False, f"{path}: not JSON: {text(out.json())[:80]!r}")
        if not formats:
            return
        schema = {
            "type": "object",
            "properties": {"capital": {"type": "string"}, "country": {"type": "string"}},
            "required": ["capital", "country"],
            "additionalProperties": False,
        }
        body = {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prose}],
            "format": schema,
            "options": {"num_predict": 200, "temperature": 0},
        }
        answer = c.post("/api/chat", json=body).json()["message"]["content"]
        try:
            keys = set(json.loads(answer))
        except ValueError:
            keys = set()
        expect(keys == {"capital", "country"}, f"schema not followed: {answer[:80]!r}")

    part("format json" if formats else "format refused (cannot constrain)", format_json)
    part(
        "ollama chat",
        lambda: expect(
            _paris(
                c.post(
                    "/api/chat", json={"model": model, "stream": False, "messages": USER}
                ).json()["message"]["content"]
            ),
            "no Paris",
        ),
    )

    def openai_stream() -> None:
        text = "".join(
            json.loads(x[6:])["choices"][0]["delta"].get("content") or ""
            for x in c.post(
                "/v1/chat/completions", json={"model": model, "messages": USER, "stream": True}
            ).text.splitlines()
            if x.startswith("data: {") and json.loads(x[6:])["choices"]
        )
        expect(_paris(text), text[:80])

    part("openai stream", openai_stream)
    part(
        "anthropic",
        lambda: expect(
            _paris(
                c.post(
                    "/v1/messages", json={"model": model, "max_tokens": 50, "messages": USER}
                ).json()["content"][0]["text"]
            ),
            "no Paris",
        ),
    )

    def tools() -> None:
        out = c.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "tools": [TOOL],
                "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
                # Greedy: a 0.5B model at the default temperature sometimes
                # garbles the call's JSON, and the check must not be a coin toss.
                "temperature": 0,
            },
        ).json()
        expect(out["choices"][0]["finish_reason"] == "tool_calls", out["choices"][0]["message"])

    part("tool call", tools)
    if logprobs:
        part(
            "logprobs",
            lambda: expect(
                c.post(
                    "/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": USER,
                        "logprobs": True,
                        "top_logprobs": 2,
                        "temperature": 0,
                    },
                ).json()["choices"][0]["logprobs"]["content"],
                "none",
            ),
        )
    else:
        refused = c.post(
            "/v1/chat/completions", json={"model": model, "messages": USER, "logprobs": True}
        )
        part(
            "logprobs refused clearly",
            lambda: expect(refused.status_code == 400, refused.status_code),
        )

    def counted() -> None:
        body = {"model": model, "max_tokens": 1, "messages": USER}
        n = c.post("/v1/messages/count_tokens", json=body)
        real = c.post("/v1/messages", json=body).json()["usage"]["input_tokens"]
        expect(n.status_code == 200 and n.json()["input_tokens"] == real, (n.text[:80], real))

    part("count_tokens", counted)


@check("E1", "llama.cpp in process (HFL_NUM_PARALLEL=1)")
def e1(a: Audit) -> str:
    # The default serves GGUF through llama-server when it is on PATH (E16);
    # one request at a time keeps llama-cpp-python in process.
    part = Parts()
    with a.server(env={"HFL_NUM_PARALLEL": "1"}) as base:
        suite(part, base, "chat")
        log = max((a.work / "logs").glob("serve-*.log"), key=lambda p: p.stat().st_mtime)
        part(
            "served in process, not by llama-server",
            lambda: expect(
                "llama-server serving" not in log.read_text(errors="replace"), "llama-server"
            ),
        )
        c = httpx.Client(base_url=base, timeout=600)
        off = c.post(
            "/api/chat",
            json={
                "model": "think",
                "stream": False,
                "think": False,
                "messages": [{"role": "user", "content": "17*23?"}],
            },
        ).json()
        part(
            "think off: answers without reasoning",
            lambda: expect(
                "391" in off["message"]["content"] and "<think>" not in off["message"]["content"],
                off["message"],
            ),
        )
    return part.verdict()


@check("E2", "llama-server (--parallel)")
def e2(a: Audit) -> str:
    need_llama_server()
    part = Parts()
    with a.server("--parallel", "4") as base:
        suite(part, base, "chat")
        c = httpx.Client(base_url=base, timeout=600)
        long = [{"role": "user", "content": "Count from 1 to 200, one number per line."}]
        body = {
            "model": "chat",
            "stream": False,
            "messages": long,
            "options": {"num_predict": 256, "temperature": 0},
        }
        c.post("/api/chat", json=body)
        singles = []
        for _ in range(3):  # the median of three: one timing was noise-bound
            started = time.monotonic()
            c.post("/api/chat", json=body)
            singles.append(time.monotonic() - started)
        one = sorted(singles)[1]
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            codes = list(pool.map(lambda _: c.post("/api/chat", json=body).status_code, range(4)))
        four = time.monotonic() - started
        part("4 at once, all answered", lambda: expect(codes == [200] * 4, codes))
        part(
            "a long reply to measure",
            lambda: expect(one > 0.3, f"one reply took {one:.2f}s: too short to measure"),
        )
        # Serialized, four take ~4x one; overlapping, less. Measured with
        # llama-server alone on a 4-core CPU (no GPU): -np 1 gave 4.00-4.15x,
        # -np 4 gave 2.85-3.18x; HFL in front added nothing measurable. The
        # 3x this used to demand sat inside that CPU's own spread; 3.5x is
        # between the two.
        part(
            "4 at once overlap (< 3.5x one)",
            lambda: expect(four < one * 3.5, f"one {one:.2f}s, four {four:.2f}s"),
        )
    return part.verdict()


def _llama_server_has_gpu(exe: str) -> bool:
    """Read here, not through HFL: the floors must not trust the code they
    check."""
    listing = subprocess.run(
        [exe, "--list-devices"], capture_output=True, text=True, timeout=60
    ).stdout
    return any(gpu in listing for gpu in ("MTL", "CUDA", "ROCm", "Vulkan", "SYCL"))


@check("E16", "parallel by default: 4 requests, two models", needs=("A24",))
def e16(a: Audit) -> str:
    """``hfl serve`` with no option serves GGUF through llama-server with 4
    slots, and it pays: ``scripts/bench_concurrency.py`` with floors.

    Measured (qwen2.5 0.5B Q4 ``chat`` + Q8 ``chat8``): M3 Max, 4 same
    model 1.88-1.93x, two models 1.44-1.51x; 4-core CPU, 1.23-1.28x and
    1.02x — there two models take turns on the cores (0.2x when they did
    not). The floors sit between those and serving in turn (1.0x) or the
    CPU collapse (0.2x)."""
    exe = need_llama_server()
    gpu = _llama_server_has_gpu(exe)
    floors = ("1.5", "1.2") if gpu else ("1.1", "0.8")
    part = Parts()
    out_file = a.work / "logs" / "concurrency.json"
    with a.server() as base:
        done = subprocess.run(
            [
                a.python, str(REPO / "scripts" / "bench_concurrency.py"), base,
                "--model", "chat", "--other-model", "chat8", "--runs", "3", "--tokens", "96",
                "--min-parallel-speedup", floors[0], "--min-two-model-speedup", floors[1],
                "--out", str(out_file),
            ],
            capture_output=True, text=True, timeout=3600,
        )  # fmt: skip
        log = max((a.work / "logs").glob("serve-*.log"), key=lambda p: p.stat().st_mtime)
        text = log.read_text(errors="replace")
        part(
            "served by llama-server, 4 slots",
            lambda: expect("shared by 4 parallel slots" in text, "no llama-server in log"),
        )
    part(
        f"speed-ups over the floors ({'GPU' if gpu else 'CPU'}: {floors[0]}x, {floors[1]}x)",
        lambda: expect(done.returncode == 0, done.stderr.strip()[-400:] or done.stdout[-400:]),
    )
    if out_file.is_file():
        d = json.loads(out_file.read_text())
        return (
            f"{part.verdict()}; 4 same {d['parallel']['speedup']}x, "
            f"two models {d['two_models']['speedup']}x"
        )
    return part.verdict()


@check("E3", "MLX")
def e3(a: Audit) -> str:
    need_apple_silicon("MLX")
    part = Parts()
    with a.server() as base:
        suite(part, base, "mlxq")
        log = max((a.work / "logs").glob("serve-*.log"), key=lambda p: p.stat().st_mtime)
        part(
            "served by MLX",
            lambda: expect("MLX" in log.read_text(errors="replace"), "no MLX in log"),
        )
    return part.verdict()


@check("E4", "Transformers")
def e4(a: Audit) -> str:
    part = Parts()
    out = a.cli(
        "pull",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--format",
        "safetensors",
        "--alias",
        "hfq",
        timeout=1800,
    )
    expect(out.returncode == 0, (out.stdout + out.stderr)[-300:])
    with a.server("--backend", "transformers") as base:
        suite(part, base, "hfq", logprobs=False)
    return part.verdict()


@check("E5", "vLLM")
def e5(a: Audit) -> str:
    """The shared chat suite on vLLM (Linux + NVIDIA), and four requests at
    once all answered — vLLM batches them. vLLM refuses logprobs and does
    not constrain formats: both must be refused clearly, not ignored."""
    import sys

    if not sys.platform.startswith("linux") or shutil.which("nvidia-smi") is None:
        raise Uncheckable("vLLM needs Linux with an NVIDIA GPU (CUDA)")
    has_vllm = subprocess.run([a.python, "-c", "import vllm"], capture_output=True, timeout=300)
    if has_vllm.returncode != 0:
        raise Uncheckable("vLLM is not installed in the audited venv (pip install 'hfl[vllm]')")
    part = Parts()
    with a.server(env={"HFL_LLM_LIBRARY": "vllm"}, ready=600) as base:
        suite(part, base, "hfq", logprobs=False, formats=False)
        body = {"model": "hfq", "stream": False, "messages": USER, "options": {"num_predict": 32}}
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            codes = list(
                pool.map(
                    lambda _: httpx.post(base + "/api/chat", json=body, timeout=600).status_code,
                    range(4),
                )
            )
        part("4 at once, all answered", lambda: expect(codes == [200] * 4, codes))
        log = max((a.work / "logs").glob("serve-*.log"), key=lambda p: p.stat().st_mtime)
        part(
            "served by vLLM",
            lambda: expect("vllm" in log.read_text(errors="replace").lower(), "no vLLM in log"),
        )
    return part.verdict()


def _gpu_mib() -> list[int]:
    """Memory in use on each NVIDIA GPU, MiB, in index order."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    return [int(x) for x in out.stdout.split()]


def _latest_serve_log(a: Audit) -> str:
    log = max((a.work / "logs").glob("serve-*.log"), key=lambda p: p.stat().st_mtime)
    return log.read_text(errors="replace")


def _answers(base: str, model: str) -> None:
    body = {"model": model, "stream": False, "messages": USER, "options": {"num_predict": 16}}
    out = httpx.post(base + "/api/chat", json=body, timeout=900)
    expect(out.status_code == 200 and _paris(out.json()["message"]["content"]), out.text[:200])


@check("E17", "several NVIDIA GPUs", needs=("E4",))
def e17(a: Audit) -> str:
    """One model over two GPUs, each way HFL offers: llama.cpp in process and
    llama-server (HFL_TENSOR_SPLIT, HFL_SPLIT_MODE + HFL_MAIN_GPU) and vLLM
    (HFL_TENSOR_PARALLEL_SIZE). In process, HFL's own log names the devices
    that took weights; llama-server's does not, so each GPU's memory before
    and after the load says where the model went."""
    import sys

    if not sys.platform.startswith("linux") or shutil.which("nvidia-smi") is None:
        raise Uncheckable("needs Linux with NVIDIA GPUs")
    if len(_gpu_mib()) < 2:
        raise Uncheckable(f"needs two NVIDIA GPUs; this machine has {len(_gpu_mib())}")
    part = Parts()
    one_at_a_time = {"HFL_NUM_PARALLEL": "1"}

    def in_process(env: dict, want: set[str]) -> None:
        with a.server(env={**one_at_a_time, **env}) as base:
            _answers(base, "chat")
            line = next((x for x in _latest_serve_log(a).splitlines() if "Acceleration:" in x), "")
        where = line.split("GiB in")[-1] if "GiB in" in line else ""
        found = set(re.findall(r"(CUDA\d)", where))
        expect(found == want, f"weights in {sorted(found) or 'no GPU'}: {line[-120:]}")

    part("llama.cpp, HFL_TENSOR_SPLIT=1,1: both GPUs",
         lambda: in_process({"HFL_TENSOR_SPLIT": "1,1"}, {"CUDA0", "CUDA1"}))  # fmt: skip
    # 1,1 is also llama.cpp's own default (layers over every GPU): 0,1 is what
    # shows HFL passes the split at all.
    part("llama.cpp, HFL_TENSOR_SPLIT=0,1: GPU 1 only",
         lambda: in_process({"HFL_TENSOR_SPLIT": "0,1"}, {"CUDA1"}))  # fmt: skip
    only_gpu1 = {"HFL_SPLIT_MODE": "none", "HFL_MAIN_GPU": "1"}
    part("llama.cpp, HFL_SPLIT_MODE=none HFL_MAIN_GPU=1: GPU 1 only",
         lambda: in_process(only_gpu1, {"CUDA1"}))  # fmt: skip

    def server(env: dict) -> list[int]:
        before = _gpu_mib()
        with a.server("--backend", "llama-server", env=env) as base:
            _answers(base, "chat")
            after = _gpu_mib()
        return [max(0, x - y) for x, y in zip(after, before)]

    def split() -> None:
        gain = server({"HFL_TENSOR_SPLIT": "1,1"})
        expect(min(gain[:2]) > 150 and min(gain[:2]) > 0.4 * max(gain[:2]), f"MiB gained {gain}")

    def main_gpu() -> None:
        gain = server({"HFL_SPLIT_MODE": "none", "HFL_MAIN_GPU": "1"})
        expect(gain[1] > 2 * max(gain[0], 1), f"MiB gained {gain}")

    part("llama-server, HFL_TENSOR_SPLIT=1,1: both GPUs", split)

    def split_to_gpu1() -> None:
        gain = server({"HFL_TENSOR_SPLIT": "0,1"})
        expect(gain[1] > 2 * max(gain[0], 1), f"MiB gained {gain}")

    part("llama-server, HFL_TENSOR_SPLIT=0,1: GPU 1", split_to_gpu1)
    part("llama-server, HFL_SPLIT_MODE=none HFL_MAIN_GPU=1: GPU 1", main_gpu)

    def vllm() -> None:
        has_vllm = subprocess.run([a.python, "-c", "import vllm"], capture_output=True, timeout=300)
        if has_vllm.returncode != 0:
            raise Uncheckable("vLLM is not installed in the audited venv")
        env = {"HFL_LLM_LIBRARY": "vllm", "HFL_TENSOR_PARALLEL_SIZE": "2"}
        with a.server(env=env, ready=600) as base:
            _answers(base, "hfq")
            used = _gpu_mib()
        expect(min(used[:2]) > 1024, f"MiB in use {used}")

    part("vLLM, HFL_TENSOR_PARALLEL_SIZE=2: both GPUs", vllm)
    return part.verdict()


def _need_nvidia() -> None:
    import sys

    if not sys.platform.startswith("linux") or shutil.which("nvidia-smi") is None:
        raise Uncheckable("needs Linux with an NVIDIA GPU")


@check("E18", "automatic backend choice on NVIDIA")
def e18(a: Audit) -> str:
    """No backend named: a safetensors model goes to Transformers on CUDA —
    not 4-bit, not the CPU — and a GGUF model to the GPU. Every other GPU
    check names its backend, so what a user gets by default was unchecked."""
    _need_nvidia()
    part = Parts()
    with a.server() as base:
        before = sum(_gpu_mib())
        part("GGUF answers", lambda: _answers(base, "chat"))
        gguf_gain = sum(_gpu_mib()) - before
        part("GGUF on the GPU", lambda: expect(gguf_gain > 200, f"GPU MiB gained {gguf_gain}"))
        part("safetensors answers", lambda: _answers(base, "hfq"))
        loaded = [x for x in _latest_serve_log(a).splitlines() if "Model loaded in" in x]
        part(
            "safetensors on CUDA (Transformers)",
            lambda: expect(loaded and " on cuda" in loaded[-1], loaded[-1:] or "no load line"),
        )
    return part.verdict()


@check("E19", "vLLM beside embeddings on one GPU", needs=("E5",))
def e19(a: Audit) -> str:
    """vLLM reserved 90 % of the GPU whatever else ran, and HFL charged it
    its weights alone: the rest looked like overhead HFL could not free, so
    an embedding model beside it (a RAG setup) could be refused. vLLM is
    now held to HFL's budget and charged what it reserves; both answer,
    side by side or with vLLM unloaded and loaded again."""
    _need_nvidia()
    part = Parts()
    with a.server(env={"HFL_LLM_LIBRARY": "vllm"}, ready=600) as base:
        part("vLLM answers", lambda: _answers(base, "hfq"))
        for model in ("embed", "minilm"):

            def embed(model: str = model) -> None:
                out = httpx.post(
                    base + "/api/embed", json={"model": model, "input": "hello"}, timeout=900
                )
                expect(out.status_code == 200 and out.json().get("embeddings"), out.text[:200])

            part(f"{model} embeds beside it", embed)
        part("vLLM answers again", lambda: _answers(base, "hfq"))
        log = _latest_serve_log(a)
    evicted = "vLLM unloaded to make room" if "make room" in log else "side by side"
    return f"{part.verdict()} ({evicted})"


@check("E20", "GPU memory as HFL plans it")
def e20(a: Audit) -> str:
    """Three GGUF models through llama-server, one after another: after each
    load, the GPU memory HFL planned ("after load") is what nvidia-smi then
    shows. llama-server runs in a child process; counted as another
    program's, each model was charged twice and the plan overshot."""
    _need_nvidia()
    part = Parts()
    with a.server() as base:
        for model in ("chat", "think", "vision"):

            def load(model: str = model) -> None:
                # Loaded and answering; what it says is not the point here
                # (a reasoning model spends 16 tokens thinking).
                httpx.post(
                    base + "/api/chat",
                    json={"model": model, "stream": False, "messages": USER,
                          "options": {"num_predict": 16}},
                    timeout=900,
                ).raise_for_status()  # fmt: skip
                real = sum(_gpu_mib()) / 1024
                plans = [x for x in _latest_serve_log(a).splitlines() if "Loading " in x]
                planned = [float(v) for v in re.findall(r"after load ~([\d.]+) GB", plans[-1])]
                expect(len(planned) >= 2, f"no GPU plan in: {plans[-1][-200:]}")
                gpu_plan = planned[-1]
                off = abs(gpu_plan - real)
                expect(
                    off <= max(0.6, 0.3 * real),
                    f"planned {gpu_plan:.1f} GB, nvidia-smi {real:.1f} GB",
                )

            part(f"{model}: planned GPU memory is what it took", load)
        ps = httpx.get(base + "/api/ps").json()["models"]
        part("all three stay resident", lambda: expect(len(ps) == 3, [m["name"] for m in ps]))
    return part.verdict()


@check("E6", "embeddings on each engine")
def e6(a: Audit) -> str:
    part = Parts()
    pulled = a.cli(
        "pull",
        "sentence-transformers/all-MiniLM-L6-v2",
        "--format",
        "safetensors",
        "--alias",
        "minilm",
        timeout=1800,
    )
    part(
        "pull a Hub embedding model (sentence-similarity)",
        lambda: expect(pulled.returncode == 0, pulled.stdout[-200:]),
    )

    def unit(base: str, model: str, size: int) -> None:
        vectors = httpx.post(
            base + "/api/embed",
            json={"model": model, "input": ["a cat", "a kitten", "tax"]},
            timeout=600,
        ).json()["embeddings"]
        norms = [round(math.sqrt(sum(x * x for x in v)), 3) for v in vectors]
        dot = lambda u, v: sum(x * y for x, y in zip(u, v))  # noqa: E731
        expect(
            norms == [1.0] * 3
            and len(vectors[0]) == size
            and dot(vectors[0], vectors[1]) > dot(vectors[0], vectors[2]),
            (norms, len(vectors[0])),
        )

    with a.server() as base:
        part("llama.cpp (GGUF nomic)", lambda: unit(base, "embed", 768))
        part("Transformers (MiniLM safetensors)", lambda: unit(base, "minilm", 384))

    def through_llama_server() -> None:
        # The install without llama-cpp-python (as Homebrew's) embeds GGUF
        # through llama-server: ``--setup``'s ``venv-core``.
        need_llama_server()
        core = a.work / "venv-core" / "bin" / "hfl"
        if not core.exists():
            raise Uncheckable("no <work>/venv-core: run with --setup")
        saved, a.hfl = a.hfl, str(core)
        try:
            with a.server() as base:
                unit(base, "embed", 768)
        finally:
            a.hfl = saved

    part("llama-server (no llama-cpp-python)", through_llama_server)
    return part.verdict()


def _wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / w.getframerate()


@check("E7", "TTS (Bark)")
def e7(a: Audit) -> str:
    part = Parts()
    pulled = a.cli("pull", "suno/bark-small", "--alias", "bark", timeout=3600)
    expect(pulled.returncode == 0, (pulled.stdout + pulled.stderr)[-300:])
    with a.server() as base:
        c = httpx.Client(base_url=base, timeout=900)
        speech = c.post("/api/tts", json={"model": "bark", "text": "Hello from the audit."})
        part(
            "POST /api/tts: a WAV of speech",
            lambda: expect(
                speech.status_code == 200 and _wav_seconds(speech.content) > 0.5,
                (speech.status_code, speech.text[:120]),
            ),
        )
    return part.verdict()


@check("E13", "a timed-out generation stops")
def e13(a: Audit) -> str:
    """Past HFL_GENERATION_TIMEOUT the client gets a 504; the generation
    must stop too, not run on to num_predict holding the model (it did,
    measured: the next request waited 16 s behind a 5 s budget)."""
    part = Parts()
    # A completion that keeps counting: a chat reply may stop by itself.
    long = {
        "model": "chat",
        "stream": False,
        "raw": True,
        "prompt": "Count from 1 to 3000, one number per line:\n1\n2\n3\n",
        "options": {"num_predict": 6000, "temperature": 0},
    }
    short = {"model": "chat", "stream": False, "messages": USER, "options": {"num_predict": 4}}
    env = {"HFL_GENERATION_TIMEOUT": "4"}

    def in_process() -> None:
        with a.server(env=env) as base:
            httpx.post(base + "/api/chat", json=short, timeout=300)  # load first
            first = httpx.post(base + "/api/generate", json=long, timeout=300).status_code
            started = time.monotonic()
            httpx.post(base + "/api/chat", json=short, timeout=300)
            waited = time.monotonic() - started
        expect(first == 504 and waited < 2.0, f"504? {first}; next request waited {waited:.1f}s")

    def llama_server() -> None:
        need_llama_server()
        log = a.home / "logs" / "llama-server-qwen2.5-0.5b-instruct-q4_k_m.log"
        before = log.read_text(errors="replace").count("cancel task") if log.exists() else 0
        with a.server("--backend", "llama-server", env=env) as base:
            httpx.post(base + "/api/chat", json=short, timeout=300)
            first = httpx.post(base + "/api/generate", json=long, timeout=300).status_code
            time.sleep(1)
        after = log.read_text(errors="replace").count("cancel task") if log.exists() else 0
        expect(
            first == 504 and after > before,
            f"504? {first}; llama-server cancelled: {after - before}",
        )

    part("llama.cpp: the next request is not kept waiting", in_process)
    part("llama-server: the task is cancelled (its log says so)", llama_server)
    return part.verdict()


COQUI_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"  # Apache-2.0, as its hifigan vocoder


@check("E12", "TTS (Coqui)", needs=(EXTRA_ID["coqui"],))
def e12(a: Audit) -> str:
    """Speech from coqui-tts, in the [coqui] extra's own venv (section C built
    it). Importing was not enough: it imported nothing under transformers 5."""
    python = a.work / "extras" / "coqui" / "bin" / "python"
    if not python.exists():
        raise Uncheckable("the [coqui] extra's venv is missing (its C check did not install it)")
    code = (
        "from hfl.engine.coqui_engine import CoquiEngine\n"
        "from hfl.engine.base import TTSConfig\n"
        "e = CoquiEngine(); e.load(%r, progress_bar=False)\n"
        "r = e.synthesize('Hello from the audit.', TTSConfig())\n"
        "print('SECONDS', r.duration)\n"
    ) % COQUI_MODEL
    env = {**a.env, "TTS_HOME": str(a.work / "tts_home")}  # its models stay in <work>
    out = subprocess.run(
        [str(python), "-c", code],
        capture_output=True,
        text=True,
        timeout=1800,
        env=env,
        cwd=a.scratch,
    )
    lines = [x for x in out.stdout.splitlines() if x.startswith("SECONDS")]
    seconds = float(lines[0].split()[1]) if lines else 0.0
    expect(seconds > 0.5, (out.stdout + out.stderr)[-300:])
    return f"{seconds:.1f}s of speech from {COQUI_MODEL}"


@check("B33", "POST /api/tts")
def b33(a: Audit) -> str:
    with a.server() as base:
        out = httpx.post(
            base + "/api/tts", json={"model": "bark", "text": "One two three."}, timeout=900
        )
        expect(
            out.status_code == 200 and _wav_seconds(out.content) > 0.5,
            (out.status_code, out.text[:200]),
        )
        missing = httpx.post(base + "/api/tts", json={"model": "nope", "text": "x"}, timeout=60)
        expect(missing.status_code == 404, missing.status_code)
    return f"{_wav_seconds(out.content):.1f}s of audio; a missing model 404"


@check("B34", "GET /api/tts/voices")
def b34(a: Audit) -> str:
    with a.server() as base:
        out = httpx.get(base + "/api/tts/voices", params={"model": "bark"}, timeout=300)
        expect(out.status_code == 200 and out.json(), out.text[:200])
    return out.text[:100]


@check("B52", "GET /v1/audio/models")
def b52(a: Audit) -> str:
    with a.server() as base:
        out = httpx.get(base + "/v1/audio/models", timeout=60)
        expect(out.status_code == 200 and "bark" in out.text, out.text[:200])
    return out.text[:100]


@check("B53", "POST /v1/audio/speech")
def b53(a: Audit) -> str:
    with a.server() as base:
        out = httpx.post(
            base + "/v1/audio/speech",
            json={"model": "bark", "input": "Hello there.", "response_format": "wav"},
            timeout=900,
        )
        expect(
            out.status_code == 200 and _wav_seconds(out.content) > 0.5,
            (out.status_code, out.text[:200]),
        )
    return f"{_wav_seconds(out.content):.1f}s WAV"


@check("A38", "hfl tts")
def a38(a: Audit) -> str:
    target = a.scratch / "tts.wav"
    a.ok("tts", "bark", "Hello from the command line.", "-o", str(target), timeout=900)
    seconds = _wav_seconds(target.read_bytes())
    expect(seconds > 0.5, seconds)
    a.fails_cleanly("tts", "no-such-model", "x")
    return f"{seconds:.1f}s WAV written; a missing model refused"


@check("A36", "hfl speak")
def a36(a: Audit) -> str:
    out = a.cli("speak", "bark", "Audit.", timeout=900)
    expect(out.returncode == 0, (out.stdout + out.stderr)[-300:])
    return "synthesised and played through the speakers"


SENTENCE = "The quick brown fox jumps over the lazy dog."


def _speech_file(a: Audit) -> Path:
    """Real speech from the system's own synthesiser: macOS ``say``, or
    ``espeak-ng`` / ``espeak`` elsewhere."""
    wav = a.scratch / "speech.wav"
    if shutil.which("say") and shutil.which("afconvert"):
        aiff = a.scratch / "speech.aiff"
        subprocess.run(["say", "-o", str(aiff), SENTENCE], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(aiff), str(wav)], check=True
        )
        return wav
    for tool in ("espeak-ng", "espeak"):
        if shutil.which(tool):
            subprocess.run([tool, "-w", str(wav), SENTENCE], check=True)
            return wav
    raise Uncheckable("no speech synthesiser to make test audio (macOS `say`, or espeak-ng)")


@check("B32", "POST /api/transcribe")
def b32(a: Audit) -> str:
    wav = _speech_file(a)
    with a.server() as base:
        out = httpx.post(
            base + "/api/transcribe",
            files={"file": ("s.wav", wav.read_bytes(), "audio/wav")},
            data={"model": "tiny"},
            timeout=900,
        )
        expect(
            out.status_code == 200 and "fox" in out.text.lower(), (out.status_code, out.text[:200])
        )
    return out.json().get("text", "")[:80]


@check("B54", "POST /v1/audio/transcriptions")
def b54(a: Audit) -> str:
    wav = _speech_file(a)
    with a.server() as base:
        out = httpx.post(
            base + "/v1/audio/transcriptions",
            files={"file": ("s.wav", wav.read_bytes(), "audio/wav")},
            data={"model": "tiny"},
            timeout=900,
        )
        expect(
            out.status_code == 200 and "fox" in out.text.lower(), (out.status_code, out.text[:200])
        )
    return out.json().get("text", "")[:80]


def _from(a: Audit, ids: list[str], what: str) -> str:
    """A summary check that holds only if the checks it summarises did."""
    results = json.loads((a.work / "results.json").read_text())
    states = {i: results.get(i, {}).get("status", "not run") for i in ids}
    expect(all(v == "OK" for v in states.values()), f"{what}: {states}")
    return f"{what}: {', '.join(ids)} OK"


@check("E8", "STT (Whisper)", needs=("B32", "B54"))
def e8(a: Audit) -> str:
    return _from(a, ["B32", "B54"], "faster-whisper tiny transcribed synthesised speech")


IMAGE_REPO = "hf-internal-testing/tiny-stable-diffusion-torch"  # Apache-2.0, 10 MB: plumbing only


@check("B15", "POST /api/images/generate")
def b15(a: Audit) -> str:
    with a.server() as base:
        out = httpx.post(
            base + "/api/images/generate",
            json={"model": IMAGE_REPO, "prompt": "a red cube", "size": "64x64", "steps": 2},
            timeout=900,
        )
        expect(out.status_code == 200, (out.status_code, out.text[:300]))
    return out.text[:120]


@check("B58", "POST /v1/images/generations")
def b58(a: Audit) -> str:
    with a.server() as base:
        out = httpx.post(
            base + "/v1/images/generations",
            json={"model": IMAGE_REPO, "prompt": "a red cube", "size": "64x64", "n": 1},
            timeout=900,
        )
        expect(out.status_code == 200 and out.json().get("data"), (out.status_code, out.text[:300]))
    return "an image returned (a 10 MB test pipeline: plumbing, not quality)"


@check("E9", "image generation (diffusers)", needs=("B15", "B58"))
def e9(a: Audit) -> str:
    return _from(a, ["B15", "B58"], "a 10 MB test pipeline (plumbing, not quality)")


def _entry(a: Audit, name: str) -> dict:
    data = json.loads((a.home / "models.json").read_text())
    models = data if isinstance(data, list) else data.get("models", [])
    return next((m for m in models if name in (m.get("name"), m.get("alias"))), {})


@check("E10", "conversion safetensors → GGUF")
def e10(a: Audit) -> str:
    part = Parts()
    repo = "HuggingFaceTB/SmolLM2-135M-Instruct"
    out = a.cli("pull", repo, "-q", "Q4_K_M", "--format", "gguf", timeout=3600)
    text = out.stdout + out.stderr
    tail = " ".join(text.split())[-200:]

    def honours_format() -> None:  # on Apple Silicon MLX used to keep safetensors
        expect("Converting to GGUF" in text, f"--format gguf ignored: {tail}")

    def converts() -> None:
        # No cmake needed any more: the converter is Python, and quantizing
        # uses a llama-quantize already here (Homebrew's, llama-cpp-python's).
        expect(out.returncode == 0 and "Traceback" not in text, tail)
        data = json.loads((a.home / "models.json").read_text())
        models = data if isinstance(data, list) else data.get("models", [])
        converted = [m for m in models if m.get("repo_id") == repo and m.get("format") == "gguf"]
        expect(converted, f"no GGUF registered for {repo}: {tail}")

    part("--format gguf honoured", honours_format)
    part("converts and quantizes, registered as GGUF", converts)
    return part.verdict()


@check("E11", "speculative decoding (DRAFT)")
def e11(a: Audit) -> str:
    modelfile = a.scratch / "DraftModelfile"
    modelfile.write_text("FROM chat\nDRAFT prompt-lookup\n")
    with a.server() as base:
        created = a.cli(
            "create", "drafted", "-f", str(modelfile), "--port", str(a.port), timeout=300
        )
        text = created.stdout + created.stderr
        expect(
            _entry(a, "drafted"), f"not created (exit {created.returncode}): {text.strip()[-200:]}"
        )
        httpx.post(
            base + "/api/chat",
            json={"model": "drafted", "stream": False, "messages": USER},
            timeout=600,
        )
        log = max((a.work / "logs").glob("serve-*.log"), key=lambda p: p.stat().st_mtime).read_text(
            errors="replace"
        )
        expect(
            "prompt-lookup" in log or "speculative" in log.lower(), "created, but no draft at load"
        )
    return "created with DRAFT; the draft used at load"


COPY = (
    "Copy this code exactly, nothing else:\n"
    "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n"
)


def _greedy(base: str, model: str) -> str:
    reply = httpx.post(
        base + "/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": COPY}],
            "options": {"temperature": 0, "num_predict": 60},
        },
        timeout=600,
    ).json()
    return str(reply.get("message", {}).get("content", reply))


def _create(a: Audit, name: str, modelfile: str) -> None:
    path = a.scratch / f"{name}.Modelfile"
    path.write_text(modelfile)
    done = a.cli("create", name, "-f", str(path), "--port", str(a.port), timeout=300)
    expect(_entry(a, name), f"{name} not created: {(done.stdout + done.stderr).strip()[-200:]}")


@check("E14", "speculative decoding on llama-server (DRAFT)")
def e14(a: Audit) -> str:
    """A DRAFT reaches llama-server and is used: its log counts accepted
    draft tokens (a draft loaded but unused — llama-server's default
    --spec-type none — shows none), and greedy output is unchanged."""
    need_llama_server()
    part = Parts()
    with a.server(env={"HFL_LLM_LIBRARY": "llama-server"}) as base:
        _create(a, "srv-draft", "FROM chat\nDRAFT chat\n")
        _create(a, "srv-lookup", "FROM chat\nDRAFT prompt-lookup\n")
        plain = _greedy(base, "chat")
        logs = sorted((a.home / "logs").glob("llama-server-*.log"))
        expect(logs, "no llama-server log")

        def accepted(model: str) -> None:
            log = max(logs, key=lambda p: p.stat().st_mtime)
            start = len(log.read_text(errors="replace"))
            text = _greedy(base, model)
            deadline = time.monotonic() + 10  # timings are written as the slot is released
            while True:
                new = log.read_text(errors="replace")[start:]
                found = re.search(r"draft acceptance = [\d.]+ \(\s*(\d+) accepted", new)
                if found or time.monotonic() > deadline:
                    break
                time.sleep(0.5)
            expect(found and int(found.group(1)) > 0, f"no draft tokens accepted: {new[-300:]}")
            expect(text == plain, f"output changed: {text[:60]!r} vs {plain[:60]!r}")

        part("a draft model", lambda: accepted("srv-draft"))
        part("prompt lookup", lambda: accepted("srv-lookup"))
    for name in ("srv-draft", "srv-lookup"):
        a.cli("rm", name, "--yes")
    return part.verdict()


@check("E15", "speculative decoding on MLX (DRAFT)")
def e15(a: Audit) -> str:
    """An MLX model with an MLX DRAFT (the same model, 4-bit): tokens come
    from the draft, greedy output is unchanged on this prompt (mlx-lm's own
    speculative decoding can drift from plain greedy on others, format or
    not — measured), and a JSON schema still holds with the draft on."""
    need_apple_silicon("MLX")
    with a.server() as base:
        _create(a, "mlx-draft", "FROM hfq\nDRAFT mlxq\n")
        plain = _greedy(base, "hfq")
        log = max((a.work / "logs").glob("serve-*.log"), key=lambda p: p.stat().st_mtime)
        start = len(log.read_text(errors="replace"))
        text = _greedy(base, "mlx-draft")
        new = log.read_text(errors="replace")[start:]
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
            "required": ["city", "country"],
            "additionalProperties": False,
        }
        formatted = httpx.post(
            base + "/api/chat",
            json={
                "model": "mlx-draft",
                "stream": False,
                "format": schema,
                "messages": [{"role": "user", "content": "Where is the Eiffel Tower?"}],
                "options": {"temperature": 0, "num_predict": 200},
            },
            timeout=600,
        ).json()["message"]["content"]
    a.cli("rm", "mlx-draft", "--yes")
    try:
        keys = set(json.loads(formatted))
    except ValueError:
        keys = set()
    expect(keys == {"city", "country"}, f"schema with a draft: {formatted[:80]!r}")
    found = re.search(r"speculative: (\d+) of (\d+) tokens from the draft", new)
    expect(found and int(found.group(1)) > 0, f"no tokens from the draft: {new[-300:]}")
    expect(text == plain, f"output changed: {text[:60]!r} vs {plain[:60]!r}")
    assert found is not None
    return (
        f"{found.group(1)} of {found.group(2)} tokens from the draft; output unchanged; "
        "a schema holds with the draft"
    )
