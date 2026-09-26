# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Section B: every route of the API, over HTTP, as a client calls it.

Routes that need a speech, transcription or image model are checked in
section E, with those models (``e_engines.py``).
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import socket
from pathlib import Path

import httpx
import tomllib
from local_audit import QUESTION, Audit, Parts, check, expect

REPO = Path(__file__).resolve().parents[2]
USER = [{"role": "user", "content": QUESTION}]
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
PIXEL = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d4944415478da63f8cfc0f01f0005000201e2267b8d0000000049454e44ae426082"
    )
).decode()


def _c(a: Audit) -> httpx.Client:
    return a.http(a.shared())


def _lines(response: httpx.Response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def _sse(response: httpx.Response) -> list[dict]:
    return [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")
    ]


def _paris(text: str) -> bool:
    return "paris" in (text or "").lower()


@check("B46", "GET /healthz")
def healthz(a: Audit) -> str:
    body = _c(a).get("/healthz").json()
    expect(body.get("status") in ("ok", "healthy"), body)
    return json.dumps(body)[:120]


@check("B1", "GET /")
def root(a: Audit) -> str:
    c = _c(a)
    expect(c.get("/").json().get("status") == "hfl is running", c.get("/").text[:200])
    page = c.get("/", headers={"Accept": "text/html"})
    expect("<html" in page.text.lower(), page.text[:200])
    return "JSON for clients, the chat page for a browser"


@check("B36", "GET /api/version")
def api_version(a: Audit) -> str:
    wanted = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]
    body = _c(a).get("/api/version").json()
    expect(body.get("version") == wanted, body)
    return body["version"]


@check("B31", "GET /api/tags")
def tags(a: Audit) -> str:
    models = _c(a).get("/api/tags").json()["models"]
    names = {m["name"] for m in models}
    expect({"chat", "embed"} & names or any("qwen" in n for n in names), names)
    expect(all("details" in m and "size" in m for m in models), models[:1])
    return f"{len(models)} models, with details and size"


@check("B61", "GET /v1/models")
def v1_models(a: Audit) -> str:
    data = _c(a).get("/v1/models").json()["data"]
    expect(any(m["id"] in ("chat",) or "qwen" in m["id"] for m in data), data[:3])
    return f"{len(data)} models"


@check("B25", "POST /api/show")
def api_show(a: Audit) -> str:
    c = _c(a)
    part = Parts()
    body = c.post("/api/show", json={"model": "chat"}).json()
    part(
        "capabilities completion+tools",
        lambda: expect(
            {"completion", "tools"} <= set(body.get("capabilities", [])), body.get("capabilities")
        ),
    )
    part("template", lambda: expect(body.get("template"), "no template"))
    part(
        "vision capability",
        lambda: expect(
            "vision"
            in c.post("/api/show", json={"model": "vision"}).json().get("capabilities", []),
            "no vision",
        ),
    )
    part(
        "missing model 404",
        lambda: expect(c.post("/api/show", json={"model": "nope"}).status_code == 404, "not 404"),
    )
    return part.verdict()


@check("B5", "POST /api/chat")
def api_chat(a: Audit) -> str:
    c = _c(a)
    plain = c.post("/api/chat", json={"model": "chat", "stream": False, "messages": USER}).json()
    expect(_paris(plain["message"]["content"]), plain)
    streamed = _lines(c.post("/api/chat", json={"model": "chat", "messages": USER}))
    text = "".join(x["message"]["content"] for x in streamed)
    expect(_paris(text) and streamed[-1]["done"], text)
    tools = c.post(
        "/api/chat",
        json={
            "model": "chat",
            "stream": False,
            "tools": [WEATHER_TOOL],
            "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        },
    ).json()
    calls = tools["message"].get("tool_calls") or []
    expect(calls and calls[0]["function"]["name"] == "get_weather", tools["message"])
    think = c.post(
        "/api/chat",
        json={
            "model": "think",
            "stream": False,
            "think": True,
            "messages": [{"role": "user", "content": "2+2?"}],
        },
    ).json()
    expect(
        think["message"].get("thinking") and "4" in think["message"]["content"], think["message"]
    )
    look = c.post(
        "/api/chat",
        json={
            "model": "vision",
            "stream": False,
            "messages": [
                {"role": "user", "content": "What colour is this image?", "images": [PIXEL]}
            ],
        },
    )
    expect(look.status_code == 200 and look.json()["message"]["content"].strip(), look.text[:300])
    missing = c.post("/api/chat", json={"model": "nope", "messages": USER})
    expect(missing.status_code == 404, missing.status_code)
    return "answer, stream, tool call, thinking, an image; missing model 404"


@check("B14", "POST /api/generate")
def api_generate(a: Audit) -> str:
    """Every part checked, each failure listed (not only the first)."""
    c = _c(a)
    body = {"model": "chat", "prompt": QUESTION, "stream": False}
    problems, fine = [], []

    def part(name: str, ok: bool, detail: object) -> None:
        (fine if ok else problems).append(name if ok else f"{name}: {str(detail)[:120]}")

    plain = c.post("/api/generate", json=body).json()
    part("answer", _paris(plain.get("response")) and plain.get("done_reason"), plain)
    streamed = _lines(c.post("/api/generate", json={**body, "stream": True}))
    part("stream", _paris("".join(x.get("response", "") for x in streamed)), streamed[-1:])
    js = c.post(
        "/api/generate",
        json={**body, "format": "json", "prompt": "Give a JSON object with a key city."},
    ).json()
    try:
        json.loads(js["response"])
        part("format json", True, "")
    except ValueError:
        part("format json", False, f"not JSON: {js['response'][:80]!r}")
    raw = c.post(
        "/api/generate",
        json={
            "model": "chat",
            "prompt": "1, 2, 3,",
            "raw": True,
            "stream": False,
            "options": {"num_predict": 4},
        },
    ).json()
    part("raw", raw.get("response", "").strip(), raw)
    lp = c.post("/api/generate", json={**body, "options": {"logprobs": 2, "num_predict": 3}}).json()
    part("logprobs", lp.get("logprobs") and len(lp["logprobs"][0]["top_logprobs"]) == 2, lp)
    ctx = c.post("/api/generate", json={**body, "options": {"keep_context": True}}).json()
    part("context", isinstance(ctx.get("context"), list) and ctx["context"], "no context")
    expect(not problems, "; ".join(problems) + f" (fine: {', '.join(fine)})")
    return ", ".join(fine)


@check("B55", "POST /v1/chat/completions")
def v1_chat(a: Audit) -> str:
    c = _c(a)
    plain = c.post("/v1/chat/completions", json={"model": "chat", "messages": USER}).json()
    expect(
        _paris(plain["choices"][0]["message"]["content"]) and plain["usage"]["total_tokens"], plain
    )
    events = _sse(
        c.post(
            "/v1/chat/completions",
            json={
                "model": "chat",
                "messages": USER,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
    )
    text = "".join(e["choices"][0]["delta"].get("content") or "" for e in events if e["choices"])
    expect(_paris(text) and events[-1].get("usage"), text)
    tools = c.post(
        "/v1/chat/completions",
        json={
            "model": "chat",
            "tools": [WEATHER_TOOL],
            "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        },
    ).json()
    expect(tools["choices"][0]["finish_reason"] == "tool_calls", tools["choices"][0])
    schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    structured = c.post(
        "/v1/chat/completions",
        json={
            "model": "chat",
            "messages": [{"role": "user", "content": "Capital of France as JSON."}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": schema},
            },
        },
    ).json()
    expect("city" in json.loads(structured["choices"][0]["message"]["content"]), structured)
    lp = c.post(
        "/v1/chat/completions",
        json={
            "model": "chat",
            "messages": USER,
            "logprobs": True,
            "top_logprobs": 2,
            "temperature": 0,
        },
    ).json()
    expect(lp["choices"][0]["logprobs"]["content"], lp["choices"][0])
    many = c.post("/v1/chat/completions", json={"model": "chat", "messages": USER, "n": 2}).json()
    expect(len(many["choices"]) == 2, many)
    bad = c.post("/v1/chat/completions", json={"model": "chat", "messages": USER, "temperature": 9})
    expect(bad.status_code in (400, 422), bad.status_code)
    return "answer, stream+usage, tool call, json_schema, logprobs, n=2; bad temperature refused"


@check("B56", "POST /v1/completions")
def v1_completions(a: Audit) -> str:
    c = _c(a)
    body = {
        "model": "chat",
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "temperature": 0,
    }
    out = c.post("/v1/completions", json=body).json()
    expect(_paris(out["choices"][0]["text"]), out)
    events = _sse(c.post("/v1/completions", json={**body, "stream": True}))
    expect(
        _paris("".join(e["choices"][0]["text"] for e in events if e.get("choices"))), events[-1:]
    )
    return "text and stream"


@check("B59", "POST /v1/messages")
def v1_messages(a: Audit) -> str:
    c = _c(a)
    body = {"model": "chat", "max_tokens": 50, "messages": USER}
    out = c.post("/v1/messages", json=body).json()
    expect(_paris(out["content"][0]["text"]) and out["stop_reason"] == "end_turn", out)
    events = _sse(c.post("/v1/messages", json={**body, "stream": True}))
    text = "".join(e["delta"].get("text", "") for e in events if e["type"] == "content_block_delta")
    expect(_paris(text) and events[-1]["type"] == "message_stop", text)
    tool = {
        "name": "get_weather",
        "description": "Current weather",
        "input_schema": WEATHER_TOOL["function"]["parameters"],
    }
    called = c.post(
        "/v1/messages",
        json={
            **body,
            "tools": [tool],
            "max_tokens": 200,
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
        },
    ).json()
    expect(called["stop_reason"] == "tool_use", called)
    thought = c.post(
        "/v1/messages",
        json={
            "model": "think",
            "max_tokens": 1500,
            "thinking": {"type": "enabled", "budget_tokens": 1000},
            "messages": [{"role": "user", "content": "2+2?"}],
        },
    ).json()
    expect([b["type"] for b in thought["content"]][:1] == ["thinking"], thought["content"])
    return "text, stream, tool_use, thinking block"


@check("B60", "POST /v1/messages/count_tokens")
def count_tokens(a: Audit) -> str:
    c = _c(a)
    body = {"model": "chat", "max_tokens": 1, "messages": USER}
    counted = c.post("/v1/messages/count_tokens", json=body).json()["input_tokens"]
    real = c.post("/v1/messages", json=body).json()["usage"]["input_tokens"]
    expect(counted == real, (counted, real))
    return f"{counted} == input_tokens of the same request"


@check("B62", "POST /v1/responses")
def v1_responses(a: Audit) -> str:
    c = _c(a)
    out = c.post("/v1/responses", json={"model": "chat", "input": QUESTION}).json()
    text = "".join(p["text"] for i in out["output"] if i["type"] == "message" for p in i["content"])
    expect(_paris(text), out)
    events = _sse(
        c.post("/v1/responses", json={"model": "chat", "input": QUESTION, "stream": True})
    )
    expect(events[-1]["type"] == "response.completed", events[-1:])
    second = c.post(
        "/v1/responses",
        json={
            "model": "chat",
            "input": "Repeat your last answer.",
            "previous_response_id": out["id"],
        },
    ).json()
    expect(second.get("previous_response_id") == out["id"], second)
    tool = {
        "type": "function",
        "name": "get_weather",
        "parameters": WEATHER_TOOL["function"]["parameters"],
    }
    called = c.post(
        "/v1/responses",
        json={"model": "chat", "tools": [tool], "input": "What's the weather in Paris?"},
    ).json()
    expect(any(i["type"] == "function_call" for i in called["output"]), called["output"])
    return "text, stream, previous_response_id, function_call"


@check("B12", "POST /api/embed")
def api_embed(a: Audit) -> str:
    c = _c(a)
    out = c.post(
        "/api/embed", json={"model": "embed", "input": ["a cat", "a kitten", "tax law"]}
    ).json()
    vectors = out["embeddings"]
    norms = [round(math.sqrt(sum(x * x for x in v)), 3) for v in vectors]
    expect(norms == [1.0, 1.0, 1.0], norms)
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))  # noqa: E731
    expect(dot(vectors[0], vectors[1]) > dot(vectors[0], vectors[2]), "no similarity order")
    wrong = c.post("/api/embed", json={"model": "chat", "input": "x"})
    expect(wrong.status_code == 400, f"a chat model: {wrong.status_code}")
    return f"unit vectors of {len(vectors[0])}; cat~kitten > cat~tax; a chat model refused"


@check("B13", "POST /api/embeddings (legacy)")
def api_embeddings(a: Audit) -> str:
    response = _c(a).post("/api/embeddings", json={"model": "embed", "prompt": "a cat"})
    expect(len(response.json()["embedding"]) == 768, response.text[:200])
    expect(response.headers.get("deprecation"), dict(response.headers))
    return "one vector; Deprecation header"


@check("B57", "POST /v1/embeddings")
def v1_embeddings(a: Audit) -> str:
    c = _c(a)
    out = c.post("/v1/embeddings", json={"model": "embed", "input": ["a", "b"]}).json()
    expect(len(out["data"]) == 2 and out["usage"]["prompt_tokens"], out)
    b64 = c.post(
        "/v1/embeddings",
        json={"model": "embed", "input": "a", "encoding_format": "base64", "dimensions": 256},
    ).json()
    raw = base64.b64decode(b64["data"][0]["embedding"])
    expect(len(raw) == 256 * 4, len(raw))
    return "float and base64; dimensions 256"


@check("B2", "POST /api/batch")
def batch(a: Audit) -> str:
    out = _c(a).post(
        "/api/batch",
        json={
            "model": "chat",
            "requests": [
                {"prompt": QUESTION, "options": {"num_predict": 8}},
                {"prompt": "2+2=", "options": {"num_predict": 4}},
            ],
        },
    )
    body = out.json()
    results = body.get("results", body)
    expect(out.status_code == 200 and len(results) == 2, out.text[:300])
    return f"{len(results)} results"


@check("B3", "POST /api/benchmark/{model}")
def benchmark(a: Audit) -> str:
    c = _c(a)
    part = Parts()
    body = {"runs_per_length": 1, "max_tokens": 8, "prompt_lengths": [16]}
    streamed = c.post("/api/benchmark/chat", json={**body, "stream": True})
    part(
        "stream: runs and summary",
        lambda: expect('"tps_mean"' in streamed.text, streamed.text[:200]),
    )
    plain = c.post("/api/benchmark/chat", json={**body, "stream": False})
    part(
        "stream false: the figures", lambda: expect("tps" in plain.text, f"only {plain.text[:120]}")
    )
    return part.verdict()


@check("B4", "POST /api/blobs/{digest}")
def blobs(a: Audit) -> str:
    c = _c(a)
    data = b"hfl audit blob"
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    ok = c.post(f"/api/blobs/{digest}", content=data)
    expect(ok.status_code in (200, 201), ok.text[:200])
    head = c.head(f"/api/blobs/{digest}")
    expect(head.status_code == 200, f"HEAD after upload: {head.status_code}")
    wrong = c.post("/api/blobs/sha256:" + "0" * 64, content=data)
    expect(wrong.status_code == 400, f"a wrong digest: {wrong.status_code}")
    return "uploaded, HEAD finds it, a wrong digest refused"


@check("B7", "POST /api/copy")
def copy(a: Audit) -> str:
    c = _c(a)
    expect(
        c.post("/api/copy", json={"source": "chat", "destination": "b-copy"}).status_code == 200,
        "copy",
    )
    expect(c.post("/api/show", json={"model": "b-copy"}).status_code == 200, "copy not found")
    expect(
        c.post("/api/copy", json={"source": "nope", "destination": "x"}).status_code == 404,
        "missing",
    )
    return "copied; missing source 404"


@check("B9", "DELETE /api/delete")
def delete(a: Audit) -> str:
    c = _c(a)
    expect(
        c.request("DELETE", "/api/delete", json={"model": "b-copy"}).status_code == 200, "delete"
    )
    expect(c.post("/api/show", json={"model": "b-copy"}).status_code == 404, "still there")
    expect(c.request("DELETE", "/api/delete", json={"model": "b-copy"}).status_code == 404, "twice")
    return "deleted; a missing model 404"


@check("B8", "POST /api/create")
def create(a: Audit) -> str:
    c = _c(a)
    part = Parts()
    out = c.post(
        "/api/create",
        json={
            "model": "b-french",
            "from": "chat",
            "system": "Answer only in French.",
            "stream": False,
        },
    )
    part("created", lambda: expect(out.status_code == 200, out.text[:200]))

    def system_applied() -> None:
        reply = c.post(
            "/api/chat",
            json={
                "model": "b-french",
                "stream": False,
                "messages": [{"role": "user", "content": "Say hello."}],
            },
        ).json()
        text = reply["message"]["content"].lower()
        expect("bonjour" in text or "salut" in text, f"system not applied: {text[:80]}")

    part("its system applies", system_applied)
    c.request("DELETE", "/api/delete", json={"model": "b-french"})
    bad = c.post("/api/create", json={"model": "b-bad", "from": "nope", "stream": False})
    part("missing base refused", lambda: expect(bad.status_code in (400, 404), bad.status_code))
    return part.verdict()


@check("B20", "GET /api/ps")
def api_ps(a: Audit) -> str:
    c = _c(a)
    c.post(
        "/api/generate",
        json={"model": "chat", "prompt": "hi", "stream": False, "options": {"num_predict": 1}},
    )
    models = c.get("/api/ps").json()["models"]
    entry = next(
        (m for m in models if "chat" in (m["name"], m.get("model"))), models[0] if models else None
    )
    expect(entry and entry.get("size") and entry.get("expires_at"), models)
    return f"{len(models)} loaded, with size and expiry"


@check("B30", "POST /api/stop")
def api_stop(a: Audit) -> str:
    c = _c(a)
    c.post(
        "/api/generate",
        json={"model": "chat", "prompt": "hi", "stream": False, "options": {"num_predict": 1}},
    )
    out = c.post("/api/stop", json={"model": "chat"})
    expect(out.status_code == 200, out.text[:200])
    names = [m["name"] for m in c.get("/api/ps").json()["models"]]
    expect("chat" not in names, names)
    return "unloaded"


@check("B35", "POST /api/verify/{model}")
def api_verify(a: Audit) -> str:
    out = _c(a).post("/api/verify/chat")
    expect(out.status_code == 200, out.text[:300])
    body = out.json()
    expect(all(c.get("passed") for c in body.get("checks", [])) or body.get("passed"), body)
    return json.dumps(body)[:120]


@check("B6", "GET /api/compliance/dashboard")
def compliance(a: Audit) -> str:
    out = _c(a).get("/api/compliance/dashboard")
    expect(out.status_code == 200 and "apache" in out.text.lower(), out.text[:300])
    return "licenses listed"


@check("B10", "GET /api/discover")
def api_discover(a: Audit) -> str:
    out = _c(a).get("/api/discover", params={"q": "qwen", "page_size": 3})
    expect(out.status_code == 200 and "qwen" in out.text.lower(), out.text[:300])
    return "Hub results"


@check("B11", "GET /api/draft/recommend")
def api_draft(a: Audit) -> str:
    out = _c(a).get("/api/draft/recommend", params={"model": "Qwen/Qwen2.5-7B-Instruct"})
    expect(out.status_code == 200 and "qwen" in out.text.lower(), out.text[:300])
    return out.text[:120]


@check("B24", "GET /api/recommend")
def api_recommend(a: Audit) -> str:
    out = _c(a).get("/api/recommend", params={"task": "chat", "top_n": 3})
    expect(out.status_code == 200 and "/" in out.text, out.text[:300])
    return out.text[:120]


@check("B21", "POST /api/pull")
def api_pull(a: Audit) -> str:
    c = _c(a)
    out = c.post(
        "/api/pull",
        json={"model": "nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0", "stream": False},
        timeout=1800,
    )
    expect(out.status_code == 200 and out.json().get("status") == "success", out.text[:300])
    missing = c.post("/api/pull", json={"model": "hfl-audit/does-not-exist-9f2c", "stream": False})
    expect(missing.status_code in (400, 404), missing.status_code)
    return "pulled over the API; a missing repo refused"


@check("B22", "POST /api/pull/smart")
def api_pull_smart(a: Audit) -> str:
    out = _c(a).post(
        "/api/pull/smart",
        json={"model": "Qwen/Qwen2.5-0.5B-Instruct", "stream": False},
        timeout=1800,
    )
    expect(out.status_code == 200, out.text[:300])
    return out.text[:120]


@check("B23", "POST /api/push")
def api_push(a: Audit) -> str:
    out = _c(a).post("/api/push", json={"model": "chat", "stream": False})
    expect(out.status_code in (400, 401, 403) and "Traceback" not in out.text, out.text[:300])
    raise PermissionError(
        f"publishing to the Hub is outward and the owner's act; without a token: {out.status_code} "
        f"{out.text[:100]}"
    )


@check("B16", "GET /api/lora")
def api_lora_all(a: Audit) -> str:
    out = _c(a).get("/api/lora")
    expect(out.status_code == 200, out.text[:200])
    return out.text[:100]


@check("B17", "POST /api/lora/apply")
def api_lora_apply(a: Audit) -> str:
    c = _c(a)
    adapter = a.home / "adapters" / "moe_shakespeare15M.gguf"
    expect(adapter.exists(), "adapter not downloaded (section A, hfl lora)")
    prompt = {
        "model": "stories",
        "prompt": "Look in thy glass",
        "raw": True,
        "stream": False,
        "options": {"temperature": 0, "top_k": 1, "num_predict": 30, "repeat_penalty": 1.0},
    }
    before = c.post("/api/generate", json=prompt).json()["response"]
    applied = c.post("/api/lora/apply", json={"model": "stories", "lora_path": str(adapter)})
    expect(applied.status_code == 200, applied.text[:200])
    after = c.post("/api/generate", json=prompt).json()["response"]
    expect(after != before, "no effect")
    a._lora = (applied.json()["adapter_id"], before, prompt)  # type: ignore[attr-defined]
    outside = c.post("/api/lora/apply", json={"model": "stories", "lora_path": "/etc/passwd"})
    expect(outside.status_code == 400, outside.status_code)
    return "applied (text changed); a path outside HFL 400"


@check("B19", "GET /api/lora/{model}")
def api_lora_model(a: Audit) -> str:
    out = _c(a).get("/api/lora/stories").json()
    expect(out.get("adapters"), out)
    return f"{len(out['adapters'])} adapter(s)"


@check("B18", "POST /api/lora/remove")
def api_lora_remove(a: Audit) -> str:
    c = _c(a)
    adapter_id, before, prompt = a._lora  # type: ignore[attr-defined]
    out = c.post("/api/lora/remove", json={"model": "stories", "adapter_id": adapter_id})
    expect(out.status_code == 200, out.text[:200])
    expect(c.post("/api/generate", json=prompt).json()["response"] == before, "not the original")
    again = c.post("/api/lora/remove", json={"model": "stories", "adapter_id": adapter_id})
    expect(again.status_code == 404, again.status_code)
    return "removed (original text); twice 404"


@check("B28", "POST /api/snapshot/save")
def snapshot_save(a: Audit) -> str:
    c = _c(a)
    c.post(
        "/api/generate",
        json={"model": "chat", "prompt": "hi", "stream": False, "options": {"num_predict": 1}},
    )
    out = c.post("/api/snapshot/save", json={"model": "chat", "name": "b-snap"})
    expect(out.status_code == 200, out.text[:200])
    return out.text[:100]


@check("B26", "GET /api/snapshot")
def snapshot_list(a: Audit) -> str:
    out = _c(a).get("/api/snapshot")
    expect("b-snap" in out.text, out.text[:200])
    return "listed"


@check("B27", "POST /api/snapshot/load")
def snapshot_load(a: Audit) -> str:
    c = _c(a)
    out = c.post("/api/snapshot/load", json={"model": "chat", "name": "b-snap"})
    expect(out.status_code == 200, out.text[:200])
    missing = c.post("/api/snapshot/load", json={"model": "chat", "name": "nope"})
    expect(missing.status_code == 404, missing.status_code)
    return "loaded; missing 404"


@check("B29", "DELETE /api/snapshot/{name}")
def snapshot_delete(a: Audit) -> str:
    c = _c(a)
    expect(c.delete("/api/snapshot/b-snap").status_code == 200, "delete")
    expect(c.delete("/api/snapshot/b-snap").status_code == 404, "twice")
    return "deleted; twice 404"


@check("B37", "POST /api/web_fetch")
def web_fetch(a: Audit) -> str:
    c = _c(a)
    out = c.post("/api/web_fetch", json={"url": "https://example.com"})
    expect(out.status_code == 200 and "example domain" in out.text.lower(), out.text[:300])
    internal = c.post("/api/web_fetch", json={"url": "http://127.0.0.1:1/"})
    expect(internal.status_code in (400, 403), f"a loopback URL: {internal.status_code}")
    return "fetched a page; a loopback URL refused"


@check("B38", "POST /api/web_search")
def web_search(a: Audit) -> str:
    out = _c(a).post(
        "/api/web_search", json={"query": "hugging face transformers", "max_results": 3}
    )
    results = out.json().get("results") if out.status_code == 200 else None
    expect(
        results,
        f"{out.status_code} {out.text[:120]} — an empty list with 200, not an error "
        "(DuckDuckGo answered 202, its bot challenge, when fetched directly)",
    )
    return f"{len(results)} results (DuckDuckGo)"


@check("B41", "GET /health")
def health(a: Audit) -> str:
    out = _c(a).get("/health").json()
    expect(out.get("status"), out)
    return json.dumps(out)[:120]


@check("B42", "GET /health/deep")
def health_deep(a: Audit) -> str:
    out = _c(a).get("/health/deep")
    expect(out.status_code == 200, out.text[:200])
    return out.text[:120]


@check("B43", "GET /health/live")
def health_live(a: Audit) -> str:
    expect(_c(a).get("/health/live").status_code == 200, "not 200")
    return "200"


@check("B44", "GET /health/ready")
def health_ready(a: Audit) -> str:
    out = _c(a).get("/health/ready")
    expect(out.status_code == 200, out.text[:200])
    return out.text[:120]


@check("B45", "GET /health/sli")
def health_sli(a: Audit) -> str:
    out = _c(a).get("/health/sli")
    expect(out.status_code == 200, out.text[:200])
    return out.text[:120]


@check("B47", "GET /metrics")
def metrics(a: Audit) -> str:
    text = _c(a).get("/metrics").text
    for name in ("hfl_requests_total", "hfl_tokens_generated_total", "hfl_model_loads_total"):
        expect(name in text, f"{name} missing")
    loads = int(
        next(
            line.split()[-1]
            for line in text.splitlines()
            if line.startswith("hfl_model_loads_total")
        )
    )
    expect(loads > 0, "hfl_model_loads_total 0 after loads")
    return f"Prometheus text; model loads {loads}"


@check("B48", "GET /metrics/json")
def metrics_json(a: Audit) -> str:
    out = _c(a).get("/metrics/json").json()
    expect(out, out)
    return ", ".join(list(out)[:5])


@check("B39", "GET /docs")
def docs(a: Audit) -> str:
    expect("swagger" in _c(a).get("/docs").text.lower(), "not the Swagger page")
    return "Swagger UI"


@check("B40", "GET /docs/oauth2-redirect")
def oauth2(a: Audit) -> str:
    expect(_c(a).get("/docs/oauth2-redirect").status_code == 200, "not 200")
    return "200"


@check("B49", "GET /openapi.json")
def openapi(a: Audit) -> str:
    paths = _c(a).get("/openapi.json").json()["paths"]
    expect(len(paths) > 40, len(paths))
    return f"{len(paths)} paths described"


@check("B50", "GET /redoc")
def redoc(a: Audit) -> str:
    expect("redoc" in _c(a).get("/redoc").text.lower(), "not ReDoc")
    return "ReDoc"


@check("B51", "GET /ui")
def ui(a: Audit) -> str:
    out = _c(a).get("/ui")
    expect("<html" in out.text.lower(), out.text[:200])
    csp = out.headers.get("content-security-policy", "")
    expect("nonce-" in csp, f"CSP: {csp}")
    return "the chat page, with a nonce CSP"


@check("B63", "WS /ws/chat")
def ws_chat(a: Audit) -> str:
    from websockets.sync.client import connect

    base = a.shared().replace("http://", "ws://")
    with connect(base + "/ws/chat", open_timeout=30) as ws:
        ws.send(json.dumps({"type": "ping"}))
        expect(json.loads(ws.recv(timeout=30))["type"] == "pong", "no pong")
        ws.send(json.dumps({"type": "chat", "model": "chat", "messages": USER}))
        frames = []
        while True:
            frame = json.loads(ws.recv(timeout=120))
            frames.append(frame)
            if frame["type"] in ("done", "error"):
                break
        text = "".join(f.get("delta", "") for f in frames if f["type"] == "token")
        expect(frames[-1]["type"] == "done" and _paris(text), frames[-3:])
    return "pong; ready, tokens, done over one connection"


def _lan_ip() -> str | None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("192.0.2.1", 9))  # no packet is sent
            return str(sock.getsockname()[0])
        except OSError:
            return None
