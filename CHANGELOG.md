# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.3] - 2026-04-17

Operational hardening release. Closes remaining issues from the
architecture review, removes dead code paths that had accumulated in
`api/helpers.py` during the 0.3.x line, and adds a request-body-size
guard that was previously missing.

### Security

- **Request body size limit** (`src/hfl/api/middleware.py`,
  `src/hfl/api/server.py`, `src/hfl/config.py`). Added
  `RequestBodyLimitMiddleware` which rejects requests whose
  `Content-Length` exceeds `config.max_request_bytes` (default 10 MiB,
  override with `HFL_MAX_REQUEST_BYTES=<bytes>`; set to `0` to
  disable). The limit runs before auth and rate-limit so oversized
  payloads are rejected with `413 PAYLOAD_TOO_LARGE` without consuming
  auth work or rate-limit tokens. Previously a malicious client could
  send a multi-gigabyte prompt and force the server to buffer it.

### Fixed

- **Python 3.16 compatibility — `asyncio.iscoroutinefunction`
  deprecation** (`src/hfl/utils/retry.py`, `src/hfl/core/tracing.py`).
  Switched both call sites to `inspect.iscoroutinefunction`. The
  asyncio wrapper is slated for removal in 3.16 and emitted
  `DeprecationWarning` on Python 3.14+, polluting test output and
  posing a near-term upgrade blocker.
- **Stale `@pytest.mark.xfail` marker on `_has_cuda`**
  (`tests/test_selector.py`). The marker was added when pytest's full
  suite re-imported torch in a way that broke the check; the
  underlying issue is long-gone and the test has been XPASSing for
  releases. Removed the marker so CI stays strict about regressions
  here.

### Removed

- **Dead model-loading helpers in `api/helpers.py`**
  (`src/hfl/api/helpers.py`, `tests/test_helpers.py`,
  `tests/test_routes_native.py`). `ensure_llm_loaded`,
  `ensure_tts_loaded`, and `run_async_with_timeout` were superseded by
  the locked `ServerState.ensure_llm_loaded` / `.ensure_tts_loaded`
  methods and `api/model_loader.py` several releases ago, but the old
  copies and their tests lingered. The old helpers had two subtle
  issues vs. the replacements: no per-model locking, and
  `engine.load()` called synchronously in the event loop instead of
  via `asyncio.to_thread`. Removed to avoid future regressions from
  someone accidentally reaching for the wrong entry point. Also
  trimmed the test that patched `hfl.api.helpers.get_registry` (it
  now patches `hfl.api.model_loader.get_registry`, matching where the
  real import lives).

### Tests

- 2107 passing (was 2113 — 14 removed with the dead helpers, 8 added
  for the new body-size middleware + env-var guard).
- Coverage held at 88.8% (was 88.8%).

## [0.3.2] - 2026-04-13

Completes Gemma 4 support by teaching HFL to parse the family's
native tool-call format. Without this fix, calling a tool with
``tools=[...]`` on a Gemma 4 model returned no structured tool_calls
and the model would hallucinate a fabricated tool response in the
content field.

### Fixed

- **Gemma 4 tool calls were dropped and the model hallucinated
  results** (`src/hfl/api/tool_parsers.py`,
  `src/hfl/engine/llama_cpp.py`). Gemma 4 emits tool calls in a
  split-pipe DSL that is not JSON:
  ``<|tool_call>call:NAME{key:<|"|>string<|"|>,num:42}<tool_call|>``.
  Strings are bracketed by the dedicated ``<|"|>`` token (ID 110 in
  the Gemma 4 vocabulary), keys are bare identifiers, and numbers /
  booleans / null are written without quotes. llama-cpp-python 0.3.x
  has no parser for this format, so ``message.tool_calls`` came back
  as ``None`` and the DSL landed raw in ``content``.
  - New ``parse_gemma4`` in ``hfl.api.tool_parsers``. Splits the body
    on the ``<|"|>`` delimiter and walks alternating outside / inside
    segments so colons inside string values (e.g. URLs) don't confuse
    the bare-key transformer. Handles nested objects, numbers,
    booleans and ``null``; falls back to empty arguments rather than
    dropping a malformed call entirely so the caller can still see
    which function was invoked.
  - ``_detect_family`` now returns ``"gemma4"`` for model names
    containing ``gemma-4`` / ``gemma4`` / ``gemma 4``, and
    ``dispatch`` routes such calls through the new parser. Earlier
    Gemma versions (2, 3) are deliberately NOT routed — they use a
    different output format and would be mis-parsed.
- **Model hallucinated a tool response and kept generating after the
  call** (`src/hfl/engine/llama_cpp.py`). Without a stop token on the
  ``<tool_call|>`` marker, the model emitted ``<|tool_response>``
  tokens and synthesised a fake result. ``LlamaCppEngine`` now builds
  the ``stop`` list via a new ``_build_stop_list()`` helper that
  appends ``<tool_call|>`` whenever ``tools`` is non-empty and the
  architecture is ``gemma4``. Caller-supplied stops are preserved.
- **Channel-marker filter was eating tool payload before it reached
  the parser** (`src/hfl/engine/llama_cpp.py`). The non-streaming
  regex and the streaming state machine both used to strip
  ``<|tool>`` / ``<|tool_call>`` / ``<|tool_response>`` and their
  closers as generic "open" / "close" markers. That removed the
  envelope before ``parse_gemma4`` could see it, so routes_native's
  ``parse_tool_calls`` call site found nothing and returned zero tool
  calls even when the model had emitted one.
  - ``_GEMMA4_OPEN_MARKER`` / ``_GEMMA4_CLOSE_MARKER`` regexes now
    exclude the tool group entirely (``<|channel|turn>`` only).
  - ``_GEMMA4_STREAM_MARKERS`` drops the tool entries so the stream
    filter lets them pass through as plain text (character-by-
    character fallthrough) and the route can re-parse the accumulated
    stream at ``done: true``.

### Tests

- 10 new tests in ``tests/test_tool_parsers.py`` covering:
  - Real-world single call captured from the bartowski GGUF during
    the incident.
  - Truncated-by-``stop`` variant where ``<tool_call|>`` was consumed
    and the regex must anchor at end-of-string.
  - Numeric / boolean / null values.
  - Nested object arguments.
  - String values with internal colons (URL regression guard).
  - Passthrough of tool-free replies.
  - Multiple tool calls in one generation.
  - Malformed body → empty arguments but function name surfaced.
  - ``dispatch`` routing by model-name substring (gemma-4 → parser,
    gemma-2 → NOT routed).

## [0.3.1] - 2026-04-13

Stability fixes for the Gemma 4 family, triggered by a real incident
where loading `gemma-4-31B-it-Q4_K_M` kernel-panicked an M3 Max twice
with a ``watchdog timeout`` (all 128 GiB of unified memory became
wired by Metal for the fp16 KV cache + flash-attn scratch of the
auto-detected 262144-token context). Also unblocks the CI type-check
job on Python 3.12.

### Fixed

- **Gemma 4 kernel panics on load** (`src/hfl/engine/llama_cpp.py`).
  Added three layers of defence in `LlamaCppEngine.load()`:
  - `_ARCHITECTURE_CTX_CAP` caps `n_ctx` to 8192 for `gemma3` and
    `gemma4` when the caller doesn't pass an explicit override,
    instead of letting llama-cpp-python auto-detect the 262144-token
    context from the GGUF header.
  - `_ARCHITECTURE_NO_FLASH_ATTN` forces `flash_attn=False` for
    `gemma4` unless the caller explicitly opts in, because
    llama-cpp-python's flash-attention Metal kernel is still
    crash-prone on this arch.
  - `_preflight_memory_check` estimates weights + KV cache against
    available system RAM and raises `OutOfMemoryError` **before** the
    `Llama` constructor runs when the load would exceed
    `_MEMORY_SAFETY_FRACTION` (85 %). The estimator is GQA-aware —
    it reads `attention.head_count` / `.head_count_kv` from the GGUF
    and uses `n_kv_heads * head_dim` instead of `embedding_length`,
    so heavy-GQA models like Gemma 4 31B (8:1 ratio) aren't falsely
    rejected. Falls back to the conservative upper bound when GQA
    metadata is missing. Set `HFL_DISABLE_MEMORY_PREFLIGHT=1` to
    bypass.
- **Gemma 4 channel markers leaking into chat output**
  (`src/hfl/engine/llama_cpp.py`). Gemma 4's native output format
  uses split-pipe reasoning delimiters (`<|channel>thought<channel|>`,
  `<|turn>...<turn|>`, `<|think>...<think|>`) and always prepends an
  empty thought channel when `enable_thinking` is False. Added a
  character-level streaming filter (`_Gemma4StreamFilter`) and its
  non-streaming counterpart (`_strip_gemma4_channel_markers`) that
  strip these markers from `chat()` and `chat_stream()` output for
  the `gemma4` architecture. The streaming filter is a state machine
  that holds the buffer until the longest possible marker match is
  determined, so markers split across token chunks (the common case
  when llama-cpp-python streams one token at a time) are still
  caught correctly. No-op for other architectures.
- **Static `_ARCHITECTURE_CHAT_FORMAT` override corrupting
  well-packaged GGUFs** (`src/hfl/engine/llama_cpp.py`). The override
  (added so community GGUFs that forget to embed
  `tokenizer.chat_template` don't fall through to llama-cpp-python's
  Llama-2 `[INST]` fallback) was also firing on bartowski / unsloth /
  lmstudio-community GGUFs that ship the correct Jinja template,
  downgrading their Gemma 4 format to Gemma 2 and silently breaking
  the prompt side. `load()` now probes the GGUF for
  `tokenizer.chat_template` and only applies the static override when
  it's missing. The embedded template is always preferred.
- **CI type-check job on Python 3.12 failing with pre-existing mypy
  errors** (`src/hfl/utils/retry.py`, `src/hfl/logging_config.py`).
  - `retry.py` dispatches to an `async_wrapper` for coroutine
    functions but the outer decorator is generic over
    `Callable[P, T]`, which mypy can't specialise to
    `Callable[P, Awaitable[T]]`. Cast `func` once inside the async
    branch so mypy can type-check the `await` correctly.
  - `logging_config.py` assigned `StructuredFormatter()` and
    `PrettyFormatter()` to the same variable in an if/else; added an
    explicit `logging.Formatter` annotation so the two branches
    don't fight over the inferred type.

### Tests

- New `tests/test_llama_cpp_preflight.py` (45 tests) covering:
  - Architecture-based `n_ctx` cap for `gemma3`/`gemma4` and explicit
    override precedence.
  - `flash_attn` disable for `gemma4` with explicit override
    precedence.
  - Preflight refusal before `Llama()` is constructed (explicit
    exploding stub verifies the constructor is never reached).
  - `HFL_DISABLE_MEMORY_PREFLIGHT` bypass.
  - `psutil`-missing graceful degradation (arch caps still apply).
  - GQA-aware vs. naive estimator with the real numbers from
    `bartowski/google_gemma-4-31B-it-GGUF` (60 layers, 5376 embed
    dim, 32/4 GQA, 262144 max context).
  - Channel marker filter in non-streaming, streaming (chunks split
    at marker boundaries), and single-character streaming modes.
  - Embedded chat_template deference vs. static override fallback.
- Project coverage rose to 89.18 % (up from 16.84 % on the paths
  this change touches), all 2131 tests pass on Python 3.10, 3.11,
  and 3.12 across Ubuntu and macOS CI matrices.

## [0.3.0] - 2026-04-13

Full implementation of the **HFL tool-calling specification**: structured
function calling over the Ollama wire protocol, an in-server inference
dispatcher with bounded queue, and operational hardening of the error
envelope / rate-limit / health surface. HFL can now drive
agent-style tool-calling loops (`write_wiki`, `read_raw`, `commit`, …)
end to end without any client-side workaround.

### Added

- **Structured tool calling** (spec §2, §4, §6)
  - `ChatRequest.tools` and `tool_choice` fields on `/api/chat`
  - `OllamaChatMessage` accepts `role="tool"`, `tool_calls`, and `name`
  - `ChatMessage` and `GenerationResult` carry `tool_calls` through the
    engine layer; `chat()` / `chat_stream()` accept a `tools=` kwarg
  - `TransformersEngine` forwards `tools` to the tokenizer's
    `apply_chat_template`, so qwen/llama3/mistral templates emit their
    native tool-call markers
  - `LlamaCppEngine` passes `tools` to `create_chat_completion` and
    normalises returned `arguments` strings to parsed dicts
  - New `hfl.api.tool_parsers` module with per-family parsers:
    - Qwen `<tool_call>...</tool_call>`
    - Llama 3 `<|python_tag|>...<|eom_id|>` and
      `<function=name>{...}</function>` (normalises `parameters` →
      `arguments`)
    - Mistral `[TOOL_CALLS][...]`
    - Generic fallback for `{"tool_call": {...}}` and
      `{"name": "...", "arguments": {...}}` envelopes
  - `/api/chat` non-streaming and streaming both emit canonical
    `message.tool_calls` with `content=""` when tools are invoked,
    matching Ollama's wire protocol

- **Inference dispatcher** (spec §5.3 — concurrency / queueing)
  - New `hfl.engine.dispatcher` module providing
    `InferenceDispatcher`, `QueueFullError`, `QueueTimeoutError`,
    `DispatcherSnapshot`
  - Bounded concurrency (`max_inflight`, default 1) with a bounded
    wait queue (`max_queued`, default 16) and acquire timeout
    (`acquire_timeout`, default 60 s)
  - Cancellation-safe: slots are always released on exception,
    timeout, or cancellation
  - Streaming endpoints pre-acquire a slot and hold it for the whole
    generator lifetime, so a long stream cannot be stamped on by a
    concurrent request
  - Global singleton wired through `hfl.core.container` and accessible
    via `hfl.core.get_dispatcher()`
  - Config: `HFL_QUEUE_ENABLED`, `HFL_QUEUE_MAX_INFLIGHT`,
    `HFL_QUEUE_MAX_SIZE`, `HFL_QUEUE_ACQUIRE_TIMEOUT`
  - All three API surfaces (Ollama, OpenAI, Anthropic) share one
    dispatcher, so a slow call on `/api/chat` correctly blocks
    `/v1/chat/completions` and `/v1/messages`

- **Operational hardening** (spec §5.1, §5.2, §5.4, §5.5)
  - `ErrorDetail` gains `category` and `retryable` fields so clients
    can decide retry vs dead-letter without parsing prose. Codes are
    mapped via `_ERROR_POLICY` and propagated by every factory and
    by `HFLHTTPException`
  - Rate-limit headers now include `X-RateLimit-Reset` (epoch) and
    `X-RateLimit-Window` (seconds) on every response; 429s return the
    structured envelope
  - New error codes: `QUEUE_FULL` (rate_limit, retryable),
    `QUEUE_TIMEOUT` (engine, retryable)
  - New `GET /healthz` endpoint returning `status`, `models_loaded`,
    `queue_depth`, `queue_in_flight`, `uptime_seconds` (200 / 503).
    `/healthz` is in `PUBLIC_ENDPOINTS` so orchestrators can always
    probe it
  - Deterministic auth on `/api/tags`: always requires the API key
    when one is configured
  - `X-Queue-Depth`, `X-Queue-In-Flight`, `X-Queue-Max-Inflight`,
    `X-Queue-Max-Size` headers on every response for live
    backpressure observability

- **Tests** — 8 new suites, 124 new tests, all green:
  - `test_schema_tool_calling.py` (13)
  - `test_engine_tools.py` (11)
  - `test_tool_parsers.py` (22)
  - `test_tool_calling_acceptance.py` (10 — spec §6 T1–T7 parametrised
    over qwen3, llama3, mistral, and the non-standard envelope)
  - `test_operational_contract.py` (7)
  - `test_dispatcher.py` (23)
  - `test_concurrency_contract.py` (11)
  - `test_concurrency_regression_safety.py` (5 — soak / stability /
    mixed endpoints / recovery after queue full)

### Changed

- `APIKeyMiddleware` 401 response now returns the structured envelope
  with `code=UNAUTHORIZED`, `category=auth`, `retryable=False`
- Rate-limit 429 response (both middleware and `rate_limit_exceeded`
  factory) now returns the structured envelope
- `CHAT_REQUEST.messages` role set expanded from
  `system|user|assistant` to `system|user|assistant|tool`
- CLI version/debug tests read `hfl.__version__` dynamically instead
  of hardcoding the version string

### Fixed

- `tests/test_cli.py::TestVersionCommand::test_version` and
  `tests/test_cli.py::TestDebugCommand::test_debug_shows_hfl_version`
  no longer regress on every version bump

### Compliance

- HFL now fulfils **every rule** of the HFL tool-calling spec
  (`hfl-tool-calling-spec.md`): C1–C8 of §2 plus §5.1–§5.5 of the
  operational section. `test_tool_calling_acceptance.py::TestT1…T7`
  act as the end-to-end acceptance suite and must stay green.

## [0.2.0] - 2026-03-18

### Added

- Anthropic Messages API compatibility (`POST /v1/messages`) so
  Claude Code and other Anthropic-SDK clients can use HFL as a
  backend
- System tray icon (`hfl[tray]` extra)
- `hfl help extras` command

### Fixed

- Rate limiter no longer leaks between tests (TTS test flakiness)
- Auto-detect context size from GGUF metadata instead of hard-coding
- Tray icon no longer stays yellow on auto-start
- CI: formatting, mypy, import ordering, starlette / python-multipart
  CVE pin-ups

## [0.1.0] - 2026-02-19

### Added

- **CLI Commands**
  - `hfl pull <model>` - Download models from HuggingFace Hub
  - `hfl run <model>` - Interactive chat with a model
  - `hfl serve` - Start API server (OpenAI + Ollama compatible)
  - `hfl list` - List downloaded models
  - `hfl search <query>` - Search HuggingFace Hub with pagination
  - `hfl inspect <model>` - Show model details
  - `hfl rm <model>` - Delete a model
  - `hfl alias <model> <alias>` - Set model aliases
  - `hfl login` / `hfl logout` - HuggingFace authentication

- **API Endpoints**
  - OpenAI-compatible: `/v1/chat/completions`, `/v1/completions`, `/v1/models`
  - Ollama-compatible: `/api/chat`, `/api/generate`, `/api/tags`

- **Inference Backends**
  - llama.cpp (GGUF models, CPU/GPU)
  - HuggingFace Transformers (safetensors, GPU)
  - vLLM (production, GPU)

- **Model Conversion**
  - Automatic safetensors to GGUF conversion
  - Quantization support (Q2_K through F16)
  - Provenance tracking for conversions

- **Legal Compliance**
  - License verification before download
  - License risk classification
  - AI output disclaimers
  - Gating system respect
  - Privacy-safe token handling

### Security

- Tokens never persisted to disk
- API server binds to localhost by default
- Network exposure requires confirmation

[0.3.0]: https://github.com/ggalancs/hfl/releases/tag/v0.3.0
[0.2.0]: https://github.com/ggalancs/hfl/releases/tag/v0.2.0
[0.1.0]: https://github.com/ggalancs/hfl/releases/tag/v0.1.0
