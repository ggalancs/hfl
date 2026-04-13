# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
