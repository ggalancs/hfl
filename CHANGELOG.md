# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
