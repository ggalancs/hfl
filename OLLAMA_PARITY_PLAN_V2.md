# OLLAMA_PARITY_PLAN_V2 — post-0.7.0 roadmap

> **Context.** `OLLAMA_PARITY_PLAN.md` took HFL from 4/16 to 16/16 on
> Ollama's documented REST endpoints (releases 0.4.0 → 0.7.0, all 8
> phases shipped, 2594 tests passing, 0 CodeQL alerts). That plan is
> complete.
>
> This document is the follow-up. It does **not** re-derive REST
> parity — the API surface is done. Instead it takes stock of
> everything *beyond* the REST surface where Ollama has moved
> forward in 2026 (v0.20.x / v0.21.0) and lays out a phased,
> auditable path to close every remaining gap — including the ones
> we've historically treated as out-of-scope.
>
> **Audience.** Architects and contributors with at least a skim of
> the first plan under their belt. Phase numbering continues from
> that plan (next phase is 9).

---

## 0. Method

Sources consulted (2026-04-17):

- `docs.ollama.com` — API, capabilities, Modelfile, context-length.
- `github.com/ollama/ollama/releases` — v0.20.0 through v0.21.0.
- HFL's own state — README, CHANGELOG.md (through 0.7.0), source
  tree under `src/hfl/`.

For every gap identified, the plan records:

1. **What Ollama does today** (with a citation where possible).
2. **What HFL does today** (or does not).
3. **Why it matters** (ecosystem / UX / performance / compliance).
4. **Scope** (files touched, estimated LOC, tests).
5. **Priority** (P0 / P1 / P2 / P3).

Every item ends up in exactly one phase. Phases are sized to fit
into a release boundary; no phase exceeds ~4 weeks of focused work.

---

## 1. Executive summary of gaps

Ollama, in 2026, is no longer just "a local model server" — it is
becoming a **local agent platform**. The REST surface was a moving
target while HFL closed the gap through Phases 1–8; since April
2026 the frontier has shifted to six areas where HFL still has
work to do:

1. **Agentic capabilities.** Web search, web fetch, MCP client +
   server, built-in agent loops.
2. **Runtime performance.** KV-cache quantisation, prefix caching,
   speculative decoding, continuous batching, VRAM-aware context
   sizing.
3. **New inference features.** Logprobs, streaming partial tool
   calls, multi-level thinking (GPT-OSS `low`/`medium`/`high`),
   batched inference, richer embedding pooling.
4. **Backend expansion.** MLX on Apple Silicon (mixed-precision
   quantisation), explicit ROCm/Vulkan paths, Whisper audio
   transcription, image generation.
5. **Modelfile as a build system.** `ENV`, `BUILD`, `INCLUDE`,
   `CAPABILITIES`, full Go-template rendering.
6. **Distribution & enterprise.** Docker images, Homebrew, Windows
   MSI, macOS notarized DMG, OAuth/SSO, OpenTelemetry,
   multi-tenancy, audit logs.

And one area where we do **not** follow Ollama even if we could: a
hosted registry (`registry.ollama.ai` analogue). HFL stays
local-first; HuggingFace Hub remains the distribution substrate.

---

## 2. Gap analysis

The matrix below is terse on purpose. Expanded discussion for each
row lives in §3 of this document.

| # | Area | Ollama 2026 | HFL 0.7.0 | Gap | Pri |
|---|------|-------------|-----------|-----|-----|
| 1 | Web search | `/api/web_search`, `/api/web_fetch` with API key | — | full endpoint + engine integration | P0 |
| 2 | MCP client | Consumes Cline/Goose/Codex MCP servers | — | protocol client | P0 |
| 3 | MCP server | Exposes its tool registry as MCP | — | protocol server | P1 |
| 4 | Agent loop | Native multi-turn tool dispatch | External only | route-level auto-resubmit of tool results | P1 |
| 5 | Streaming partial tool calls | Yes | Tool calls emitted only at end | incremental JSON chunks | P1 |
| 6 | Multi-level thinking | `low` / `medium` / `high` for GPT-OSS | bool only | thread the level through config + engine | P1 |
| 7 | Logprobs | `logprobs`, `top_logprobs` in response | — | engine → route → schema | P1 |
| 8 | `/api/batch` | Batched prompts, one request | — | new endpoint | P2 |
| 9 | KV cache quantisation | `OLLAMA_KV_CACHE_TYPE=q4_0` / `q8_0` | — | expose to config | P1 |
| 10 | Prefix caching | Yes (prompt cache reuse) | — | llama-cpp's `cache_prompt` + LRU | P1 |
| 11 | Speculative decoding | Draft-model path | — | `draft_model` in config | P2 |
| 12 | Continuous batching | Concurrent requests share KV cache | Per-slot queue | true continuous batching | P2 |
| 13 | VRAM-aware ctx sizing | 4k / 32k / 256k tiers by VRAM | Fixed `n_ctx` | auto-probe GPU memory | P1 |
| 14 | MLX backend | Mixed-precision on Apple Silicon | — | new engine module | P1 |
| 15 | ROCm/Vulkan build detection | Yes | llama-cpp inherits | build-time checks + docs | P2 |
| 16 | Whisper / audio-in | Yes (local transcription) | — | `/api/transcribe` + engine | P2 |
| 17 | Image generation | SDXL, FLUX via pipeline | — | `/api/images/generate` | P3 |
| 18 | Embedding pooling | mean / cls / last-token | mean only | config option | P1 |
| 19 | Modelfile `ENV` | Environment vars baked into manifest | — | parser + loader | P2 |
| 20 | Modelfile `BUILD` | Multi-stage builds | — | parser + executor | P3 |
| 21 | Modelfile `INCLUDE` | Import from another Modelfile | — | parser + resolver | P2 |
| 22 | Modelfile `CAPABILITIES` | Explicit capability flag list | Auto-detected | parser + manifest | P2 |
| 23 | Template — full Go rendering | Yes (native Go templates) | Regex substitution | port to a real renderer | P1 |
| 24 | OpenTelemetry | Traces, spans | — | `opentelemetry-api` wiring | P2 |
| 25 | OAuth / SSO | OpenCode integration | API-key only | OAuth middleware | P2 |
| 26 | RBAC / multi-tenant | — (own side: ollama.com) | — | per-tenant registry isolation | P3 |
| 27 | Audit logs | — | Partial via metrics | structured JSON audit trail | P2 |
| 28 | Docker image | Multi-arch CUDA + Metal | — | Dockerfile + CI | P0 |
| 29 | Homebrew formula | `brew install ollama` | — | formula PR | P1 |
| 30 | Windows MSI | WinGet + signed MSI | — | installer script | P2 |
| 31 | macOS notarized DMG | Yes | — | codesign + notarize flow | P2 |
| 32 | PyPI wheel auto-publish | — (they ship Go binary) | Manual | release GH Action | P1 |
| 33 | Interactive REPL | Rich TUI w/ history | Basic typer prompt | `prompt_toolkit` replacement | P2 |
| 34 | VS Code extension | Yes (OpenCode) | — | separate repo | P3 |
| 35 | JetBrains plugin | Yes | — | separate repo | P3 |
| 36 | Snapshot / restore | Session save/load | — | CLI + persistence layer | P3 |
| 37 | Sandboxed execution | Considered | — | seccomp / containerisation | P3 |
| 38 | Model signing | Planned | SHA-256 verify | add signature layer | P2 |
| 39 | Gemma 4 BPE + `add_bos_token` | Respected | Partial | verify + adapt engine | P1 |
| 40 | Parallel tool-call fanout | Yes | Serial per request | dispatch tools concurrently at route | P1 |

40 rows. 6 P0, 14 P1, 13 P2, 7 P3.

---

## 3. Deep-dive — per-area scope, rationale, acceptance

### 3.1 Agentic capabilities (rows 1–4, P0/P1)

The single biggest behavioural gap between HFL and Ollama in April
2026. Ollama ships agentic features first-class; HFL's agents have
to bolt everything on externally.

#### 3.1.1 — `/api/web_search` + `/api/web_fetch` (row 1, P0)

**Scope**

- New module `src/hfl/tools/web_search.py` with a pluggable backend
  (DuckDuckGo HTML scrape as free default; Tavily, Brave, SerpAPI
  as env-keyed alternatives via `HFL_WEB_SEARCH_BACKEND`).
- New routes `src/hfl/api/routes_web.py`:
  - `POST /api/web_search` body `{"query": str, "max_results": 1..10}`.
  - `POST /api/web_fetch` body `{"url": str}` returning
    `{"title": str, "content": str, "links": [str]}`.
- Rate limit per-IP (`HFL_WEB_SEARCH_RPM`), shared 10 Hz NDJSON
  progress if streaming.
- Response envelope byte-compatible with `docs.ollama.com/capabilities/web-search`:
  `{"results": [{"title","url","content"}]}`.

**Tests (≥20)**

DDG scrape parsing (6 corpora), Tavily mocked (3), backend
switching via env (2), rate-limit 429 (1), URL scheme validation
(SSRF guard), large-body fetch truncation, charset detection.

**Files touched.** 6. **LOC.** ~500. **Pri.** P0 (blocks agents).

#### 3.1.2 — MCP client (row 2, P0)

**Scope**

- Depend on the official Python SDK: `mcp[client] >= 1.0`.
- New module `src/hfl/mcp/client.py` with an async helper that
  connects to an MCP server over stdio or SSE, enumerates tools,
  and translates them into HFL's tool-calling schema.
- New CLI `hfl mcp connect <transport>://<...>`.
- New server-side option `HFL_MCP_AUTOLOAD` pointing at a JSON
  file with MCP-server configs; `hfl serve` connects on boot.
- Route `/api/chat` auto-includes MCP tools alongside any
  ``tools`` in the request body.

**Tests (≥15)**

End-to-end with a dummy MCP server (tool discovery, invocation,
error handling), auto-load config parsing, CLI round-trip,
client timeouts, transport negotiation.

**Files touched.** 10. **LOC.** ~800. **Pri.** P0.

#### 3.1.3 — MCP server (row 3, P1)

**Scope**

- Expose HFL's internal "tools" (web_search, web_fetch, model
  manipulation primitives like `model.create`, `blob.upload`) as
  an MCP server over stdio and SSE.
- Entry point `hfl mcp serve --transport {stdio,sse}`.
- Security: token-gated; each connection scoped to a single
  capability set via a manifest file.

**Tests (≥10)**

Schema export correctness, capability filtering, transport
switching, graceful shutdown.

**Files touched.** 6. **LOC.** ~500. **Pri.** P1.

#### 3.1.4 — Native agent loop (row 4, P1)

**Scope**

- New helper `src/hfl/api/agent_loop.py` that, when
  `/api/chat` receives `tool_choice="auto"` and the model emits
  tool calls, automatically invokes the resolved tools (MCP or
  web_*) and re-submits the results to the model for a follow-up
  turn. Capped by `max_iterations` (default 6) to prevent
  runaway loops.
- Returns the final assistant message with
  `tool_trace: [{"call","result"}]` for replay / debugging.
- Streaming variant emits interleaved thinking + tool_call +
  tool_result NDJSON events.

**Tests (≥12)**

3-hop loop with a deterministic fake tool, max_iterations cap,
tool-error recovery, streaming ordering, tool_choice="none"
disables the loop entirely.

**Files touched.** 5. **LOC.** ~400. **Pri.** P1.

---

### 3.2 Streaming & thinking (rows 5–6, P1)

#### 3.2.1 — Streaming partial tool calls (row 5, P1)

**Current.** HFL's tool-parser runs on the final accumulated
text; a client streaming the conversation never sees the
`tool_calls` array until the model is done generating. Ollama
since v0.20.6 emits partial JSON chunks so the client can render
progress ("calling `get_weather` with `{...`") in real time.

**Scope**

- Teach the tool parser (`src/hfl/api/tool_parsers.py`) to accept
  a streaming feed: it keeps a running buffer and yields
  `PartialToolCall(index, name_delta, arguments_delta)` objects
  as soon as enough structure is visible.
- `/api/chat` stream path attaches each partial to the NDJSON
  chunk envelope under `message.tool_calls` (list of dicts with
  only the fields present so far — Ollama's shape).

**Tests (≥8)**

Per-architecture partial parsing (Qwen, Gemma, Llama3), malformed
JSON tail recovery, stream restart idempotence.

**Files touched.** 4. **LOC.** ~300. **Pri.** P1.

#### 3.2.2 — Multi-level thinking (row 6, P1)

**Current.** `think: true|false` only. GPT-OSS and newer Gemma 4
tunes accept `low` / `medium` / `high`.

**Scope**

- Widen `GenerationConfig.expose_reasoning` → `thinking_level:
  Literal["off","low","medium","high"] = "off"`. Back-compat:
  `think=True` maps to `"medium"`, `think=False` maps to `"off"`.
- Schema (`schemas/ollama.py`) accepts `"low"|"medium"|"high"|bool`.
- Architecture filter in `llama_cpp.py` adjusts the channel
  threshold: `"low"` strips everything; `"high"` keeps the full
  channel stream; `"medium"` is the current behaviour.

**Tests (≥6)**

Level round-trip, legacy bool compatibility, per-architecture
channel selection.

**Files touched.** 3. **LOC.** ~150. **Pri.** P1.

---

### 3.3 Inference features (rows 7–8, 18, 40)

#### 3.3.1 — Logprobs (row 7, P1)

**Scope**

- `GenerationConfig.logprobs: int = 0` (0 = off, 1..20 = top-k).
- `GenerationResult.logprobs: list[dict] | None` with shape
  `[{"token": str, "logprob": float, "top_logprobs": [{...}]}]`.
- llama-cpp's sampler already exposes `logprobs`; wire via the
  `logprobs=` kwarg on `Llama.__call__`.
- OpenAI-compat (`/v1/chat/completions`) emits the OpenAI shape
  `choices[0].logprobs`.
- Ollama-native emits under `logprobs` at the envelope root.

**Tests (≥6)** shape correctness + integration.

**Files touched.** 5. **LOC.** ~250. **Pri.** P1.

#### 3.3.2 — `/api/batch` (row 8, P2)

**Scope**

- New route `POST /api/batch` body
  `{"model": str, "requests": [{"prompt": str, "options": {...}}, …]}`.
- Executes every request sequentially (v1) under the same engine
  state; streaming is opt-in per-request. Continuous batching
  (row 12) upgrades this to parallel.
- Returns `{"results": [envelope, …]}`.

**Tests (≥8)** happy path, mixed stream / non-stream rejected,
partial failure isolation.

**Files touched.** 3. **LOC.** ~200. **Pri.** P2.

#### 3.3.3 — Embedding pooling strategies (row 18, P1)

**Current.** Mean pooling only. Sentence-transformers models
frequently expect CLS pooling; Qwen3 embeddings expect
last-token.

**Scope**

- `POST /api/embed` body accepts `pooling: Literal["mean","cls",
  "last"] = "mean"`.
- Engine-level pooling helper in
  `src/hfl/engine/embedding_pooling.py`.

**Tests (≥5)** per strategy + dimensionality invariants.

**Files touched.** 4. **LOC.** ~180. **Pri.** P1.

#### 3.3.4 — Parallel tool-call dispatch (row 40, P1)

**Current.** When a turn emits multiple tool_calls HFL returns
them to the client; there's no server-side fanout.

**Scope**

- When the agent loop (3.1.4) is active, tool calls in the same
  turn run concurrently via `asyncio.gather`. Per-tool timeouts
  are honoured. Results are folded back in call-order.

**Tests (≥4)** concurrent timing, partial-failure fold-back.

**Files touched.** 1. **LOC.** ~100. **Pri.** P1.

---

### 3.4 Runtime & performance (rows 9–13, 39)

#### 3.4.1 — KV cache quantisation (row 9, P1)

**Scope**

- `config.kv_cache_type: Literal["f16","q8_0","q4_0"] = "f16"`,
  overridable per-load via the Modelfile `PARAMETER kv_cache_type`.
- LlamaCppEngine sets `type_k=`, `type_v=` on the
  `Llama()` kwargs.
- CLI `hfl serve --kv-cache-type q8_0`.

**Tests (≥5)** kwarg plumbing + config surface.

**Files touched.** 4. **LOC.** ~150. **Pri.** P1.

#### 3.4.2 — Prefix caching / prompt cache (row 10, P1)

**Scope**

- Enable llama-cpp's `cache_prompt=True` by default.
- For longer-lived reuse across requests, maintain an LRU of the
  last N prompt-prefix KV states (keyed by SHA-256 of the prompt
  prefix), capped at 2 GB by default via
  `HFL_PROMPT_CACHE_MAX_BYTES`.

**Tests (≥6)** LRU eviction, cache-hit speedup assertion
(mock-timed), config disable.

**Files touched.** 3. **LOC.** ~300. **Pri.** P1.

#### 3.4.3 — Speculative decoding (row 11, P2)

**Scope**

- `config.draft_model: str | None`. When set, engine loads a
  second (smaller) Llama as draft; main model runs speculative
  verification.
- Requires llama-cpp-python ≥ 0.3.20 which ships the
  `draft_model` API.

**Tests (≥4)** load-time resolution, regression on throughput
(mocked).

**Files touched.** 2. **LOC.** ~200. **Pri.** P2.

#### 3.4.4 — Continuous batching (row 12, P2)

**Scope**

- Upgrade the dispatcher (`src/hfl/engine/dispatcher.py`) from
  per-slot serial execution to continuous batching: a single
  dedicated worker pulls up to N inflight prompts, interleaves
  their token generation using llama-cpp's `n_batch` knob, and
  fans the NDJSON streams back to the right clients.
- Shape preserved; only throughput changes.

**Tests (≥8)** concurrency correctness under 4/16/64 inflight,
latency fairness.

**Files touched.** 6. **LOC.** ~600. **Pri.** P2.

#### 3.4.5 — VRAM-aware context sizing (row 13, P1)

**Scope**

- On model load, probe available VRAM via
  `pynvml` (CUDA), `torch.backends.mps.current_allocated_memory`
  (Metal), or `/sys/class/drm/*/mem_info_vram_used` (ROCm).
- Pick `n_ctx` from tiers: `<24 GB → 4096`, `24–48 GB → 32768`,
  `≥48 GB → 262144`. Override via
  `config.default_ctx_size` or Modelfile `PARAMETER num_ctx`.

**Tests (≥4)** tier selection, probe fallback to CPU default.

**Files touched.** 2. **LOC.** ~150. **Pri.** P1.

#### 3.4.6 — Gemma 4 BPE + `add_bos_token` (row 39, P1)

**Scope**

- Read `tokenizer.add_bos_token` from the GGUF header; when
  `False`, pass `add_bos=False` into `Llama()` and to
  `tokenize()`.
- Detect SentencePiece-BPE vs pure-SentencePiece and expose via
  `manifest.tokenizer_type`.

**Tests (≥4)** header parse, engine kwargs flow.

**Files touched.** 2. **LOC.** ~80. **Pri.** P1.

---

### 3.5 Backend expansion (rows 14–17)

#### 3.5.1 — MLX backend (row 14, P1)

**Scope**

- New module `src/hfl/engine/mlx_engine.py` implementing
  `InferenceEngine` over `mlx-lm`. Conditional import behind
  an `[mlx]` extra.
- Architecture-detection: `mlx_lm.utils.load()` handles
  Llama/Gemma/Qwen families. Unsupported families raise
  `UnsupportedModelArchitecture`.
- Mixed-precision quantisation honours `manifest.quantization`
  (`Q4`, `Q5`, `Q8`).

**Tests (≥12)** skipped automatically on non-Apple hardware; load
+ generate smoke + tokenizer roundtrip.

**Files touched.** 5. **LOC.** ~700. **Pri.** P1.

#### 3.5.2 — ROCm/Vulkan build detection (row 15, P2)

**Scope**

- `hfl doctor` command prints the detected accelerators and
  whether llama-cpp-python was built against each. Documentation
  page on choosing builds.

**Tests (≥3)** mocked `rocm-smi` / `vulkaninfo` probes.

**Files touched.** 2. **LOC.** ~150. **Pri.** P2.

#### 3.5.3 — Whisper transcription (row 16, P2)

**Scope**

- New endpoint `POST /api/transcribe` accepting multipart audio
  (≤ 100 MB), body fields `model`, `language`.
- Engine `WhisperEngine` using `openai-whisper` or
  `faster-whisper` via a new `[stt]` extra.
- CLI `hfl transcribe <file>`.

**Tests (≥6)** model loading, multipart parsing, language
override, timestamp metadata.

**Files touched.** 7. **LOC.** ~500. **Pri.** P2.

#### 3.5.4 — Image generation (row 17, P3)

**Scope**

- Endpoint `POST /api/images/generate` with body
  `{"model": str, "prompt": str, "size": "1024x1024", "steps": int}`.
- Engine `DiffusersEngine` using `diffusers` over an
  `[image-gen]` extra. Supports SDXL (base + refiner) and FLUX.
- Response stream emits base64 PNGs.

**Tests (≥8)** end-to-end mocked.

**Files touched.** 8. **LOC.** ~900. **Pri.** P3.

---

### 3.6 Modelfile as a build system (rows 19–23)

#### 3.6.1 — `ENV` directive (row 19, P2)

**Scope**

- Parser recognises `ENV KEY=VALUE` lines; stores into
  `ModelfileDocument.env: dict[str, str]`.
- Persisted on the manifest as `env_vars`.
- At engine load time, they are merged into the subprocess
  environment (for engines that shell out) and into
  `os.environ` *scoped to that load* via a context manager.

**Tests (≥6)** parser, manifest roundtrip, scoped application.

#### 3.6.2 — `BUILD` directive (row 20, P3)

**Scope**

- Multi-stage: `BUILD base …`, `BUILD converted …`; each stage
  produces an intermediate artefact with its own digest. Allows
  chaining `safetensors → gguf → quantised` without hand-running
  each step.

**Tests (≥10)** pipeline orchestration.

#### 3.6.3 — `INCLUDE` / imports (row 21, P2)

**Scope**

- `INCLUDE ./common.modelfile` inlines another Modelfile at parse
  time. Path resolution relative to the including file. Cycle
  detection.

**Tests (≥6)** cycle detection, relative paths.

#### 3.6.4 — `CAPABILITIES` (row 22, P2)

**Scope**

- `CAPABILITIES completion,tools,vision,embedding` explicit flag.
- Supersedes HFL's auto-detection (`detect_capabilities`) when
  present — human-authored Modelfiles take precedence.

**Tests (≥4)** parser + manifest override.

#### 3.6.5 — Real Go-template rendering (row 23, P1)

**Current.** HFL does a regex substitution for `{{ .Prompt }}`
and `{{ .System }}`. Ollama uses full Go templates, so things
like `{{- range .Messages }} ... {{ end -}}`, conditionals,
whitespace trimming and function calls are not supported.

**Scope**

- Depend on `gotemplate` (a pure-Python port) or translate each
  template to Jinja at load time. Prefer the port for fidelity.
- Fallback: on unsupported syntax, emit a deprecation warning
  and pass the template through literally (current behaviour).

**Tests (≥15)** Llama3 template, Gemma4 template, Qwen2.5
template, range/if/with, whitespace trimming.

**Files touched.** 3. **LOC.** ~400 (incl. the port).
**Pri.** P1 — user-visible correctness.

---

### 3.7 Enterprise & observability (rows 24–27)

#### 3.7.1 — OpenTelemetry tracing (row 24, P2)

**Scope**

- `opentelemetry-api` + `opentelemetry-sdk` as a soft-dep behind
  `[otel]` extra; config fields `otel_enabled`,
  `otel_exporter_endpoint`, `otel_service_name`.
- Spans around: request, dispatcher queue, engine call, tool
  call. Links to the NDJSON event stream via a trace-id on
  every envelope.

**Tests (≥6)** span shape, exporter mocking.

#### 3.7.2 — OAuth / SSO (row 25, P2)

**Scope**

- New middleware supporting OIDC (Google, GitHub, Keycloak). API
  tokens remain the fallback.
- Token → user → per-user quota.

**Tests (≥8)** OIDC round-trip with a mock IdP.

#### 3.7.3 — RBAC / multi-tenancy (row 26, P3)

**Scope**

- Per-tenant namespace on the registry. `ModelManifest` grows
  `owner: str | None`; routes filter by the authenticated
  identity.
- Tenant creation via `hfl admin tenant create <name>`.

**Tests (≥10)** isolation invariants + admin CLI.

#### 3.7.4 — Audit logs (row 27, P2)

**Scope**

- Structured JSON log line per privileged event (model load,
  create, delete, copy, pull, stop, keep_alive change, API key
  mint). Writable to a file via `HFL_AUDIT_LOG_PATH` with
  rotation.

**Tests (≥6)** log-line shape, rotation.

---

### 3.8 Distribution (rows 28–32)

#### 3.8.1 — Docker image (row 28, P0)

**Scope**

- Multi-arch (`linux/amd64`, `linux/arm64`, `linux/amd64-cuda`,
  `linux/arm64-cuda`) images on GHCR.
- `Dockerfile` with build args for CUDA version, ROCm version.
- GH Action: on every tag, build + push + sign with cosign.

**Tests.** CI smoke (`docker run --rm hfl version`).

#### 3.8.2 — Homebrew formula (row 29, P1)

**Scope**

- `Formula/hfl.rb` in a `homebrew-hfl` tap repo. On release, CI
  PRs the new version into the tap.

#### 3.8.3 — Windows MSI (row 30, P2)

**Scope**

- WiX toolset recipe; signed installer; WinGet manifest.

#### 3.8.4 — macOS notarized DMG (row 31, P2)

**Scope**

- `create-dmg` + `codesign` + `notarytool` + staple. Driven by a
  GH Action that owns the notarisation secrets.

#### 3.8.5 — PyPI wheel auto-publish (row 32, P1)

**Scope**

- GH Action on tag → `python -m build` → `twine upload`. Trusted
  publisher (OIDC) so no API tokens live in the repo.

---

### 3.9 UX polish (rows 33–37)

#### 3.9.1 — Interactive REPL with `prompt_toolkit` (row 33, P2)

**Scope**

- Replace the current typer prompt with a `prompt_toolkit`
  session. History file under `~/.hfl/history`. Autocompletion
  for slash commands (`/set temperature 0.7`, `/show modelfile`,
  `/save session.json`).
- Multiline editing via `Alt-Enter`.
- `/load session.json` restores messages + options.

**Tests (≥8)** completions, history persistence, /set/show.

#### 3.9.2 — VS Code extension (row 34, P3)

Published separately (`hfl-vscode`). Out of scope for this repo;
this plan only tracks the commitment.

#### 3.9.3 — JetBrains plugin (row 35, P3)

Ditto.

#### 3.9.4 — Session snapshot / restore (row 36, P3)

**Scope**

- `hfl save <name>` / `hfl load <name>` persists the chat
  history, system prompt, loaded model, and all options to
  `~/.hfl/sessions/<name>.json`.

#### 3.9.5 — Sandboxed execution (row 37, P3)

**Scope**

- `hfl serve --sandbox seccomp` enables a seccomp-bpf filter
  that blocks network egress from the inference process.
  macOS variant uses App Sandbox. Linux: landlock + seccomp.
- Excludes web_search / MCP (those are intentional egress).

---

### 3.10 Security (row 38)

#### 3.10.1 — Model signing (row 38, P2)

**Scope**

- Manifests grow `signature: str | None` (ed25519 over the
  concatenation of every blob digest).
- `hfl verify <model>` recomputes and compares against a
  trust-root public key under
  `~/.hfl/trusted-publishers.json`.
- Pull flow refuses to register a manifest that fails
  verification when strict mode is enabled.

---

## 4. Phased roadmap

Phases continue from 8. Each phase is a release-sized chunk.

| Phase | Target rel. | Scope | Duration | Pri mix |
|---|---|---|---|---|
| **9** — Web & MCP  | 0.8.0 | rows 1, 2, 28, 32 | 3 wk | 3·P0 + 1·P1 |
| **10** — Agentic  | 0.8.1 | rows 3, 4, 5, 6, 40 | 2 wk | 4·P1 + 1·P1 |
| **11** — Runtime I | 0.9.0 | rows 9, 10, 13, 23, 39 | 3 wk | all P1 |
| **12** — Inference features | 0.9.1 | rows 7, 18 | 1 wk | 2·P1 |
| **13** — Backends | 0.10.0 | rows 14, 29 | 3 wk | 2·P1 |
| **14** — Modelfile v2 | 0.10.1 | rows 19, 21, 22, 27 | 2 wk | 4·P2 |
| **15** — Runtime II | 0.11.0 | rows 8, 11, 12, 15 | 4 wk | 4·P2 |
| **16** — Audio + images | 0.12.0 | rows 16, 17 | 4 wk | 1·P2 + 1·P3 |
| **17** — Enterprise | 0.12.1 | rows 24, 25, 26, 30, 31, 33, 38 | 4 wk | mixed P2/P3 |
| **18** — Polish | 0.12.2 | rows 20, 34, 35, 36, 37 | 3 wk | mixed P3 |

**Total.** 10 phases, ~29 weeks (~7 months of focused work).
Prioritisation picks P0 first, so releases 0.8.0 and 0.8.1 alone
close the two gaps that matter most in 2026 (web/MCP/Docker +
agent loop).

---

## 5. Rationale

- **Why P0 on Docker (row 28).** Zero friction for evaluation.
  Ollama's install story is `brew install ollama` or a one-liner
  Docker pull; HFL today requires a Python env. Closing this
  unblocks adoption more than any individual feature.
- **Why P0 on web/MCP.** In 2026 agents are the gravity. An agent
  that can't search the web or consume external tools is a toy.
  These are also the features Ollama added *last*, so catching up
  now is a compressed window — the delta keeps shrinking.
- **Why Modelfile v2 is P2.** `ENV`, `INCLUDE`, `CAPABILITIES`
  matter to power users but don't unblock an ecosystem of
  clients. Deferring past the agentic block is correct.
- **Why image generation is P3.** Diffusion workflows have their
  own tooling universe (ComfyUI, Invoke, Automatic1111). HFL
  supporting `/api/images/generate` is a "check the box" feature,
  not a differentiator.
- **Why multi-tenant RBAC is P3.** HFL is local-first. The
  minority of deployments that need real multi-tenancy will want
  either a dedicated product or a Kubernetes-hosted wrapper;
  baking it into the single-process server raises operational
  complexity for everyone.

---

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| MCP protocol is still churning (breaking changes in minor versions) | Pin the SDK version, add an integration test against the reference servers; skip versions that break >3 fields |
| MLX matrix is limited (Apple Silicon only, narrower arch list) | Gate the engine behind `platform.system() == "Darwin"` + `platform.machine() == "arm64"`; fall through to llama-cpp otherwise |
| KV-cache quant (q4_0) degrades quality on smaller models | Expose the knob but default `f16`; add a doc warning |
| Continuous batching invalidates the dispatcher's current per-request timing metrics | Keep per-request timings in a thread-local; aggregate at fan-out time |
| Go-template port fidelity vs. Ollama server's Go implementation | CI corpus of 50 real Modelfiles from `ollama.com/library`; snapshot the renderer output |
| Diffusers load time + VRAM cost | Lazy-load behind the `[image-gen]` extra; no import by default |
| OpenTelemetry overhead when enabled | Zero-cost when disabled (no span creation); when enabled, batch exporter with 512 queue |
| Windows MSI signing cost (code-sign cert) | Start with an unsigned `.msi` + WinGet warning; upgrade to signed once release cadence justifies it |

---

## 7. Non-goals

These are deliberately **not** in scope for any phase of this
plan — they were evaluated and rejected or deferred with an
explicit reason.

- **`POST /api/push` + hosted registry.** HFL is local-first; HF
  Hub is the distribution substrate. Building a proprietary
  registry is a product, not a feature.
- **Cloud mode / SaaS offering.** Same reason.
- **Model pricing / metering.** Local-first means there is no
  meter to feed.
- **Ollama app (Electron GUI).** The tray app
  (`hfl[tray]`) is good enough; a full Electron frontend is
  a separate product.
- **Proprietary model catalog.** We stay on the open web; users
  use HF Hub, Modelfile `FROM`, or their own infra.

---

## 8. Success metrics

A phase lands when **all** of the following are true:

1. `scripts/ci-local.sh` exits 0 (the same gate every phase of
   `OLLAMA_PARITY_PLAN.md` cleared).
2. Coverage on code added in the phase is ≥ 90%. (Project-wide
   coverage gate stays at 75%.)
3. Ollama-SDK smoke: the Python and JS official Ollama SDKs
   exercise the phase's new endpoints without vendor-specific
   patches.
4. Zero new CodeQL alerts introduced.
5. CHANGELOG.md has a section dedicated to the release with
   every user-visible change called out.
6. README tables (English and Spanish) reflect the new coverage.

Across the whole plan the single top-line success metric is:

> **An agent stack (LangChain, LlamaIndex, Cline, Goose) pointed
> at `http://localhost:11434` should behave indistinguishably
> whether HFL or Ollama is serving — across every request shape
> documented on docs.ollama.com.**

At the end of Phase 18, that invariant should be end-to-end
provable.

---

## 9. Files this plan will touch (preview)

Top-level directories where new code lands:

- `src/hfl/tools/` — new, for web_search / web_fetch.
- `src/hfl/mcp/` — new, MCP client + server.
- `src/hfl/api/agent_loop.py` — new.
- `src/hfl/engine/mlx_engine.py` — new.
- `src/hfl/engine/whisper_engine.py` — new.
- `src/hfl/engine/diffusers_engine.py` — new.
- `src/hfl/engine/embedding_pooling.py` — new.
- `src/hfl/converter/modelfile_parser.py` — extended.
- `src/hfl/converter/go_template.py` — new (pure-python port).
- `src/hfl/observability/` — new, OTEL + audit log.
- `src/hfl/security/signing.py` — new.
- `docker/` — new, multi-arch Dockerfiles.
- `.github/workflows/release.yml` — extended, wheel publish +
  Docker push.
- `Formula/hfl.rb` — new tap repo (separate).

Expected net additions: ~8,000 LOC + ~650 tests across all
phases, for a final test count of roughly **3,250** from today's
2,594.

---

## 10. Next action

Open an RFC issue on GitHub linking to this document, request
review from the one or two people who have been touching the
inference path, and merge it only after alignment on the phase
ordering.

Don't start writing code for Phase 9 until this document is
merged — the first plan's discipline (one phase, one release,
every commit ci-local green) is what kept the 8-phase run clean
and we want the second plan to inherit that cadence.
