# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Section E: each engine, through the same HTTP checks; and the routes that
need a speech, transcription or image model (section B ids)."""

from __future__ import annotations

import concurrent.futures
import io
import json
import math
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
            ("/api/chat", {"messages": [{"role": "user", "content": prose}], "format": "json"},
             lambda r: r["message"]["content"]),
        ):  # fmt: skip
            payload = {"model": model, "stream": False, "options": {"num_predict": 200}, **body}
            out = c.post(path, json=payload)
            if not formats:
                expect(out.status_code == 400 and "constrain" in out.text, (path, out.status_code))
                continue
            expect(out.status_code == 200, (path, out.status_code, out.text[:120]))
            try:
                json.loads(text(out.json()))
            except ValueError:
                expect(False, f"{path}: not JSON: {text(out.json())[:80]!r}")

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


@check("E1", "llama.cpp in process (default)")
def e1(a: Audit) -> str:
    part = Parts()
    with a.server() as base:
        suite(part, base, "chat")
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
        started = time.monotonic()
        c.post("/api/chat", json=body)
        one = time.monotonic() - started
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            codes = list(pool.map(lambda _: c.post("/api/chat", json=body).status_code, range(4)))
        four = time.monotonic() - started
        part("4 at once, all answered", lambda: expect(codes == [200] * 4, codes))
        part(
            "a long reply to measure",
            lambda: expect(one > 0.3, f"one reply took {one:.2f}s: too short to measure"),
        )
        part(
            "4 at once overlap (< 3x one)",
            lambda: expect(four < one * 3, f"one {one:.2f}s, four {four:.2f}s"),
        )
    return part.verdict()


@check("E3", "MLX")
def e3(a: Audit) -> str:
    need_apple_silicon("MLX")
    part = Parts()
    with a.server() as base:
        suite(part, base, "mlxq", formats=False)
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
        suite(part, base, "hfq", logprobs=False, formats=False)
    return part.verdict()


@check("E5", "vLLM")
def e5(a: Audit) -> str:
    raise Uncheckable("vLLM needs Linux with an NVIDIA GPU (CUDA); no check written for it yet")


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

    def fails_cleanly_or_converts() -> None:
        expect("Traceback" not in text, f"traceback: {tail}")
        if shutil.which("cmake") is None:
            expect(out.returncode != 0 and "cmake is not installed" in text, tail)
            raise Uncheckable("the conversion needs cmake, not installed here (said so clearly)")
        data = json.loads((a.home / "models.json").read_text())
        models = data if isinstance(data, list) else data.get("models", [])
        converted = [m for m in models if m.get("repo_id") == repo and m.get("format") == "gguf"]
        expect(out.returncode == 0 and converted, tail)

    part("--format gguf honoured", honours_format)
    part("converts, or says what it lacks", fails_cleanly_or_converts)
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
