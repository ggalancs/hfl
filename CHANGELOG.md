# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-04-17

**Phase 6 of OLLAMA_PARITY_PLAN — full Modelfile support.** The
feature people asked for first: ``ollama create`` now works
byte-for-byte against HFL. Derived models can be defined
declaratively via a Modelfile, stored content-addressed via blobs,
and downloaded / templated / coerced at request time. This release
closes the last P2 gaps in Ollama REST coverage — HFL is now 16/16
on the documented endpoints.

### Added — Modelfile parser + ``/api/create`` (P2-1)

- New module ``hfl.converter.modelfile_parser``:
  - ``parse_modelfile(text) -> ModelfileDocument``: formal
    tokeniser-based parser. Recognises ``FROM``, ``PARAMETER``,
    ``TEMPLATE``, ``SYSTEM``, ``ADAPTER``, ``LICENSE``,
    ``MESSAGE``, ``REQUIRES``.
  - Triple-quoted multi-line blocks (``"""..."""``) and single-line
    quoted strings with ``\\n`` / ``\\t`` / ``\\"`` / ``\\\\``
    escapes. Parameter coercion to int / float / bool per the
    canonical Ollama taxonomy; unknown keys are preserved as
    strings.
  - ``render_modelfile_document(doc)`` round-trips parsed documents
    to deterministic text.
  - ``ModelfileParseError`` carries a 1-based ``.line`` number so
    ``/api/create`` can surface precise 400 diagnostics.
- New route ``POST /api/create``:
  - NDJSON progress stream with Ollama-parity status strings
    (``parsing modelfile`` → ``using existing layer sha256:...`` →
    ``creating model`` → ``writing manifest`` → ``success``).
  - FROM resolution supports three shapes: uploaded blob
    (``sha256:<hex>``), existing model name / alias, and local
    file path.
  - Accepts both the legacy ``modelfile`` text body and the newer
    Ollama shape of structured fields (``from``, ``system``,
    ``template``, ``license``, ``parameters``, ``adapters``,
    ``messages``, ``files``). Structured fields override Modelfile
    contents when both are present.
  - ``stream=false`` mode returns a single JSON envelope (200 on
    success, 400 on error).
- New ``ModelManifest`` fields: ``system``,
  ``default_parameters``, ``adapter_paths``, ``messages``,
  ``parent_name``, ``parent_digest``. All backwards-compatible
  with on-disk manifests from earlier versions.
- CLI: ``hfl create <name> -f path/to/Modelfile`` streams the
  server's NDJSON progress and prints each status. Matches
  ``ollama create`` flag-for-flag.

### Added — ``HEAD`` / ``POST /api/blobs/:digest`` (P2-2)

- New module ``hfl.hub.blobs`` with content-addressed storage at
  ``<home_dir>/blobs/sha256-<hex>``.
- ``write_blob_stream`` streams chunks into a temp file with a
  running SHA-256, atomically renames into place only if the
  computed digest matches. Mismatches are surfaced as 400 with the
  expected vs. computed prefixes in the error message.
- ``parse_digest`` accepts ``sha256:<hex>``, ``sha256-<hex>``, and
  bare hex; rejects anything else before ``Path.joinpath`` can see
  it (path-traversal-safe).
- ``HEAD /api/blobs/:digest`` returns 200/404/400; ``POST`` returns
  201 with an ``X-Blob-Bytes`` header on success.
- ``/api/blobs/`` is whitelisted from
  ``RequestBodyLimitMiddleware`` so multi-GB GGUFs clear the
  global text-oriented cap; the streaming hasher remains the
  authoritative size check.

### Added — ``template`` + ``raw`` per-request fields (P2-3)

- ``GenerateRequest.template``: substitutes the model's default
  chat template for this request. The llama-cpp engine does a
  conservative substitution of ``{{ .Prompt }}`` (and
  ``{{ .System }}``) so the 90% case just works without pulling
  in a full Go-template parser.
- ``GenerateRequest.raw``: bypasses all prompt shaping — neither
  the template nor the ``system`` preamble is applied. Useful for
  evaluation harnesses and manual prompt engineering.
- Both surface on ``GenerationConfig`` as
  ``template_override: str | None`` and ``raw: bool``.

### Ollama REST coverage

| # | Endpoint | Status |
|---|---|---|
| 1  | ``POST /api/generate`` | complete |
| 2  | ``POST /api/chat`` | complete |
| 3  | ``POST /api/create`` | **new** |
| 4  | ``GET /api/tags`` | complete |
| 5  | ``POST /api/show`` | complete |
| 6  | ``POST /api/copy`` | complete |
| 7  | ``DELETE /api/delete`` | complete |
| 8  | ``POST /api/pull`` | complete |
| 9  | ``POST /api/push`` | non-goal (local-first) |
| 10 | ``POST /api/embed`` | complete |
| 11 | ``GET /api/ps`` | complete |
| 12 | ``HEAD /api/blobs/:digest`` | **new** |
| 13 | ``POST /api/blobs/:digest`` | **new** |
| 14 | ``GET /api/version`` | complete |
| 15 | ``POST /api/stop`` | complete |
| 16 | ``POST /api/embeddings`` (legacy) | complete |

### Test & CI

- Tests added: 106 (53 parser + 30 blobs + 11 create + 12 template/raw).
- Total suite: 2558 passing, 28 skipped. Coverage ≈ 89%.
- No CI/CD workflows were triggered during development — the
  ``feat/ollama-parity-0.4.x`` branch is outside the ``main`` /
  ``develop`` triggers. CI fires for the first time on the PR that
  merges Phases 1-6 together.

---

## [0.5.1] - 2026-04-17

**Phase 5 of OLLAMA_PARITY_PLAN — the P1 polish items.** Three new
request fields, one new management endpoint, and real nanosecond
timings on every generation response. This is the release that
closes the gap with ``ollama-python`` for anyone keying off request
parameters (``system``, ``think``) or response metrics
(``total_duration``, ``prompt_eval_duration``, ``eval_duration``).

### Added — ``system`` on ``/api/generate`` + ``/api/chat`` (P1-1)

Override the model's default system prompt per request:

- On ``/api/generate`` the string is prepended to ``prompt`` with
  two blank lines of separation.
- On ``/api/chat`` it's inserted as the *first* message with
  ``role=system``, stacking with any other system message the
  caller already supplied (Ollama allows multiple).

### Added — ``think`` on ``/api/generate`` + ``/api/chat`` (P1-1)

Expose the model's native reasoning / chain-of-thought channel:

- ``think=False`` (default) — the Gemma-4-family channel filter
  keeps ``content`` clean; reasoning is suppressed. Unchanged
  behaviour.
- ``think=True`` — filter disabled, engine returns the raw channel
  text. The route then runs the new
  ``hfl.api.thinking.extract_thinking`` to separate reasoning
  blocks from the answer and emits them in a dedicated ``thinking``
  field on the response envelope (Ollama 2026 shape).

Recognised reasoning dialects: DeepSeek-R1 / Qwen3-Thinking
``<think>...</think>``, Gemma 4
``<|channel>thought...<channel|>`` and ``<|think>...<think|>``,
``<thinking>...</thinking>``, ``<reasoning>...</reasoning>`` (o1-
style).

### Added — ``POST /api/copy`` + ``hfl cp`` (P1-2)

Duplicate a model under a new name. The copy shares the on-disk
blob — it's a registry-level operation, not a byte copy. Status
codes match Ollama byte-for-byte: 200 on success, 404 when the
source is missing, 400 when the destination is taken.

``hfl cp <src> <dst>`` CLI mirrors ``ollama cp`` with identical
output.

### Added — ``ModelRegistry.copy(source, destination)``

Underlying primitive. Thread-safe (file lock + RLock), validates
the destination name, drops the source's alias on the copy to
avoid double-booking.

### Added — nanosecond timings on ``GenerationResult`` (P1-3)

Four new fields — ``total_duration``, ``load_duration``,
``prompt_eval_duration``, ``eval_duration`` — all in nanoseconds,
Ollama-convention. ``LlamaCppEngine.chat`` and ``generate`` now
instrument with ``time.monotonic_ns`` and apportion
prompt-eval-vs-eval time proportionally to token counts (the
llama-cpp API doesn't expose the pre-first-token delta natively).

Ollama-native routes propagate the whole block:
``total_duration``, ``load_duration``, ``prompt_eval_count``,
``prompt_eval_duration``, ``eval_count``, ``eval_duration``. Pre-
0.5.1 these fields were hard-coded to 0 — tokens/sec dashboards
now work out of the box.

### Tests (+60)

- ``tests/test_thinking.py`` (14) — every reasoning dialect
  recognised and extracted; whitespace handling; idempotent on
  clean text.
- ``tests/test_system_think_routes.py`` (12) — end-to-end
  ``system`` injection on both routes; ``think`` sets
  ``expose_reasoning``; ``thinking`` field populated when the
  model emits markers; no field added when it doesn't; passthrough
  when think=False.
- ``tests/test_routes_copy.py`` (11) — registry-level copy: new
  entry / missing source / destination collision / invalid name
  / copy-by-alias. HTTP route: 200 on success, 404 on missing
  source, 400 on conflict, 422 on empty fields.
- ``tests/test_timing_metrics.py`` (7) — defaults on
  ``GenerationResult``, ``/api/generate`` surfaces all four
  timings, ``/api/chat`` surfaces all four, sum of phases within
  5% of total, non-negativity guard.

Plus the default-case tests adjusted — existing suites that
asserted ``total_duration == 0`` no longer apply because the real
value is surfaced.

### Metrics

| | 0.5.0 | 0.5.1 |
|-|-|-|
| Tests | 2411 | 2452 (+41) |
| Ollama REST endpoints | 14/16 | 15/16 (``/api/copy`` added) |
| ``system`` override | ❌ | ✅ on generate + chat |
| ``think`` override | ❌ | ✅ with reasoning extraction |
| Response timings | hard-coded 0 | real ns from engine |

Remaining parity items: Modelfile ingestion + ``/api/create`` +
blobs, and ``/api/push`` (non-goal). Those land in 0.6.x.

## [0.5.0] - 2026-04-17 — BREAKING

**Phase 4 of OLLAMA_PARITY_PLAN — vision / multimodal.** Models like
Gemma 3, Llama 4, Qwen2-VL, LLaVA, Moondream and Pixtral can now
serve image-grounded prompts through HFL on both the Ollama-native
and OpenAI-compatible routes.

### ⚠️ Breaking: OpenAI ``content`` is now a union

``ChatCompletionMessage.content`` accepts **either** the legacy
string shape **or** OpenAI's list-of-parts shape
(``[{"type":"text","text":"..."},
{"type":"image_url","image_url":{"url":"data:image/..."}}]``). Every
existing text-only client keeps working without changes.

### Added — wire contracts

- **Ollama ``/api/chat`` messages gain ``images: list[str]``**
  (base64 or data-URI). Max 32 images per message, 20 MiB each,
  4096×4096 max dimensions.
- **OpenAI ``content: list[ContentPart]``** with ``TextContentPart``
  + ``ImageContentPart`` discriminated union. Only ``data:image/...``
  URIs are honoured — HTTP(S) URLs are rejected at the route
  boundary (SSRF guard).

### Added — validator (``hfl.api.image_validator``)

Four-gate validator, standalone-testable, no Pillow required:

1. **Size cap** (20 MiB default, configurable).
2. **Magic-byte sniffing** for PNG / JPEG / WEBP / GIF — SVG, HTML,
   executables pretending to be images are rejected.
3. **MIME whitelist** — raster formats only.
4. **Dimension parsing** via format-specific headers (PNG IHDR, JPEG
   SOF scan, WebP VP8/VP8L/VP8X, GIF screen descriptor) with a
   16-megapixel total-pixel cap that catches pathological
   aspect ratios like 8192×1.

### Added — engine support

- **``LlamaCppEngine`` vision wiring.** ``load()`` auto-detects an
  adjacent ``mmproj-*.gguf`` file and builds the right
  ``chat_handler`` (Gemma3, Qwen2.5-VL, LLaVA 1.5/1.6, Moondream).
  An explicit ``clip_model_path`` kwarg overrides the auto-scan.
  Import-fallback: on older llama-cpp-python installs without the
  ``llama_cpp.llama_chat_format`` submodule the engine logs a
  warning and loads text-only rather than crashing.
- **``_messages_to_llama_cpp``** converts ``ChatMessage.images`` into
  llama-cpp's native ``[{"type":"text"}, {"type":"image_url"}]``
  content shape, sniffs each image's MIME, and emits a correctly-
  typed ``data:`` URI — all four formats (PNG/JPEG/WEBP/GIF) are
  detected from their actual bytes.

### Added — request-to-engine translator (``hfl.api.vision``)

- ``decode_ollama_images(list[str] | None) -> list[bytes] | None``
- ``split_openai_content(str | list[ContentPart]) -> (text, images)``

Both route through ``validate_image`` so engines receive clean,
bounded bytes. Errors include the failing ``images[i]`` / 
``content[i]`` index so clients know which attachment was bad.

### Tests (+60)

- ``tests/test_image_validator.py`` (28): size gate, magic bytes
  for each supported format + SVG/HTML/ELF rejection, dimension
  caps, base64 + data-URI round-trips, configurable limits.
- ``tests/test_vision_routes.py`` (15): end-to-end on both Ollama
  ``/api/chat`` and OpenAI ``/v1/chat/completions``; image bytes
  reach the engine; multi-image ordering; invalid payload 400;
  schema cap 422; HTTP URL rejected; OpenAI text-part concatenation;
  unit tests of the split/decode helpers.
- ``tests/test_llama_cpp_vision.py`` (17): ``ChatMessage.images``
  translation, MIME-sniff-per-format, base64 round-trip, arch→handler
  dispatch (Gemma3, Gemma4, Qwen-VL, LLaVA 1.5/1.6, Moondream,
  unknown fallback), import fallback when multimodal module
  missing, end-to-end load with ``mmproj-*.gguf`` auto-detection,
  explicit ``clip_model_path`` overrides the auto-scan.

### Metrics

| | 0.4.2 | 0.5.0 |
|-|-|-|
| Tests | 2351 | 2411 (+60) |
| Coverage | ~89% | ~89% |
| Ollama message-level parity | tools + keep_alive + format | + images + content parts |
| Vision models supported | 0 | Gemma 3, Gemma 4, LLaVA 1.5/1.6, Qwen2-VL, Qwen2.5-VL, Moondream (via llama-cpp) |

Remaining parity items (system/think override, copy, Modelfile
ingestion, blobs) land in 0.5.x and 0.6.x.

## [0.4.2] - 2026-04-17

**Phase 3 of OLLAMA_PARITY_PLAN — structured outputs.** Every client
library that requests JSON-constrained generation (instructor,
LangChain OutputParsers, LlamaIndex Pydantic programs, the Anthropic
Claude SDK's ``tool_use``) now flows through HFL with zero patches.

### Added — Ollama ``format``

``POST /api/generate`` and ``POST /api/chat`` accept:

- ``format: "json"`` → free-form JSON.
- ``format: { ...JSON Schema... }`` → strict schema conformance.
- ``format: "GBNF:<body>"`` → raw GBNF grammar passthrough for
  advanced users.

The field is validated at the router boundary (depth ≤10, total
properties ≤200, regex patterns ≤1024 chars) so abusive schemas
fail fast with 400 instead of hanging the grammar compiler.

### Added — OpenAI ``response_format``

``POST /v1/chat/completions`` accepts the full OpenAI JSON-mode
envelope:

- ``{"type": "text"}`` → unconstrained.
- ``{"type": "json_object"}`` → free-form JSON.
- ``{"type": "json_schema", "json_schema": {"name": "...", "schema":
  {...}}}`` → strict schema conformance. The inner ``schema`` is
  unwrapped and validated.

### Internal

- New ``GenerationConfig.response_format`` field (``str | dict |
  None``) carries the normalised constraint from router to engine.
- ``LlamaCppEngine.chat`` maps the field to llama-cpp-python's
  native ``response_format`` (for ``json``/schema) or ``grammar``
  (for raw GBNF). Older llama-cpp versions that don't support those
  kwargs fall back gracefully — the TypeError path strips the
  kwargs and retries, so the engine still produces unconstrained
  text rather than 500ing.
- New module ``hfl.api.structured_outputs`` — normalisation for
  both Ollama and OpenAI shapes plus the depth/property/pattern
  validator. Standalone unit-testable.

### Tests

- 33 new tests across ``test_structured_outputs.py`` (schema
  validator, OpenAI/Ollama envelope normalisation) and
  ``test_routes_structured_outputs.py`` (field flows from HTTP body
  through to ``engine.chat``'s GenerationConfig).

### Metrics

| | 0.4.1 | 0.4.2 |
|-|-|-|
| Tests | 2318 | 2351 (+33) |
| Ollama ``format`` on /api/generate+/api/chat | ❌ | ✅ |
| OpenAI ``response_format`` | ❌ | ✅ |
| JSON Schema DoS guard (depth/props/patterns) | ❌ | ✅ |

Remaining parity items (P0-6 vision, Modelfile ingestion, blobs,
copy, system/think overrides, metrics) land in 0.5.x.

## [0.4.1] - 2026-04-17

**Phase 2 of OLLAMA_PARITY_PLAN is complete — embeddings are live.**
Every RAG pipeline (LangChain, LlamaIndex, ChromaDB, pgvector) that
speaks the Ollama or OpenAI embedding API can now point its base URL
at HFL and just work.

### Added — embedding endpoints

- **``POST /api/embed``** (Ollama preferred): single string or list
  of strings → ``{embeddings: [[...], ...], model, total_duration,
  load_duration, prompt_eval_count}``. Accepts ``truncate``,
  ``dimensions`` (Matryoshka), ``keep_alive``, ``options``. Input
  bounds: ≤1024 strings per batch, ≤2 MiB per string, dimensions
  1-8192.
- **``POST /api/embeddings``** (legacy alias): ``{model, prompt}``
  → ``{embedding: [...]}``. Kept for older ollama-python releases
  and some LangChain versions.
- **``POST /v1/embeddings``** (OpenAI-compatible): strict OpenAI
  envelope (``object: "list"``, ``data[].object: "embedding"``,
  ``data[].index``, ``usage.prompt_tokens``/``total_tokens``),
  supports ``encoding_format: "base64"`` via little-endian float32
  packing, accepts token-list inputs (``[int]`` / ``[[int]]``)
  with documented lossy fallback to space-separated strings.

### Added — embedding engine hierarchy

- **``hfl.engine.embedding_engine.EmbeddingEngine``** — abstract
  base with a single method: ``embed(inputs, *, truncate,
  dimensions) -> EmbeddingResult``.
- **``LlamaCppEmbeddingEngine``** — wraps llama-cpp-python with
  ``embedding=True``. Same extra, different construction mode. Auto
  native-dim detection via ``Llama.n_embd()``, char-count fallback
  for token accounting when ``tokenize`` raises.
- **``TransformersEmbeddingEngine``** — AutoModel + AutoTokenizer
  with attention-masked mean pooling and L2-normalisation, matching
  sentence-transformers defaults. Auto-selects CUDA / MPS / CPU.

### Changed

- ``SUPPORTED_MODEL_TYPES`` in ``hfl.converter.formats`` now
  includes ``ModelType.EMBEDDING``. The existing
  ``EMBEDDING_ARCHITECTURES`` table (BERT / Nomic / Jina / BGE /
  GTE / E5 / MXBAI / Stella / Arctic-Embed) now grants those models
  runnable status at ``/api/embed`` entry.

### Internal

- Embedding requests bypass the LLM dispatcher (``max_inflight=1``)
  because embeds are stateless and batchable — concurrent RAG
  queries no longer block each other.
- Model loader keeps embed engines in a dedicated ``state._embed_engine``
  slot so they don't collide with chat residency.

### Metrics

| | 0.4.0 | 0.4.1 |
|-|-|-|
| Tests | 2283 | 2318 (+35) |
| Coverage | 89.05% | 88.39% |
| Ollama REST endpoints | 10/16 | 13/16 |
| RAG frameworks working out of the box | 0 | LangChain, LlamaIndex, ChromaDB, etc. |

Remaining Ollama-parity items (P0-5 structured outputs, P0-6
vision, Modelfile ingestion, blobs, copy) land in 0.4.x and 0.5.x
releases — tracked in ``OLLAMA_PARITY_PLAN.md``.

## [0.4.0] - 2026-04-17

**Phase 1 of OLLAMA_PARITY_PLAN is complete.** HFL now speaks the
operational Ollama contract that Open WebUI, ollama-python,
LangChain and LibreChat rely on. A client pointing its
``OLLAMA_HOST`` at HFL gets the same endpoints, the same JSON
shapes, and the same CLI verbs as a real Ollama server for every
management operation (list, show, ps, pull, stop, keep_alive).

Six new REST endpoints + four new CLI commands, 78 new tests, zero
regressions on the 2137-test baseline from 0.3.5.

### Added — REST endpoints

- **``GET /api/ps``** — list running models with
  ``name``/``model``/``size``/``digest``/``details``/``expires_at``/``size_vram``.
  Reflects the server's resident LLM + TTS in the Ollama shape.
  Digest prefers ``manifest.file_hash`` and falls back to a
  deterministic hash over identity so the field is never empty.
  ``size_vram`` is sourced from ``engine.memory_used_bytes()`` when
  the backend exposes it, else from the manifest's on-disk size.
- **``POST /api/show``** — full Ollama-parity envelope:
  ``modelfile`` (rendered Modelfile text), ``parameters`` (multiline
  key/value), ``template`` (chat template), ``details``,
  ``model_info`` with GGUF-style keys (``general.architecture``,
  ``<arch>.context_length``, …), ``capabilities`` and ``license``.
  Unknown model → 404 ``ModelNotFoundError``.
- **``POST /api/pull``** — real endpoint (previously only the CLI
  had this). NDJSON progress stream matching Ollama's sequence:
  ``pulling manifest`` → ``downloading`` (with heartbeats) →
  ``verifying sha256 digest`` → ``writing manifest`` →
  ``success``. ``stream=false`` returns a single JSON envelope;
  errors collapse to ``{"status":"error","error":"..."}``.
- **``POST /api/stop``** — graceful unload by name, or for all
  resident engines when ``model`` is omitted. Unload runs in a
  background task so the HTTP response is not gated on teardown.
  Idempotent (second stop returns ``{"status":"not_loaded"}``);
  clears the keep_alive deadline as a side effect.

### Added — request fields

- ``keep_alive`` on ``/api/generate`` and ``/api/chat``. Accepts
  every Ollama-compatible form: ``"5m"``, ``"30s"``, ``"1h30m"``,
  raw numbers (``10`` / ``10.0``), ``0`` (unload after this request),
  ``-1`` (keep loaded indefinitely), ``null`` (default). Malformed
  values fail fast with 400 before the dispatcher is engaged.
  ``keep_alive=0`` schedules a background unload via
  ``state.cleanup`` so the event loop stays responsive.
  ``/api/ps``'s ``expires_at`` field (R13) lights up from the
  resulting deadline.

### Added — CLI

- ``hfl ps`` — NAME / ID / SIZE / PROCESSOR / UNTIL table,
  column-for-column parity with ``ollama ps``.
- ``hfl show <model>`` — summary panel with architecture,
  parameters, quantization, format, context, size, capabilities,
  license. Flags ``--modelfile`` / ``--parameters`` / ``--template``
  / ``--license`` scope the output to a single section, like
  ``ollama show --<section>``.
- ``hfl stop [model]`` — unload one model, or everything when the
  argument is omitted. Connects to the running server over HTTP;
  helpful error when the server is not up.

### Added — capability detector

- ``src/hfl/models/capabilities.py`` maps a manifest to Ollama's
  taxonomy: ``completion``, ``tools``, ``insert``, ``vision``,
  ``embedding``, ``thinking``. Parametrised tests cover 30+ real
  model names across the Qwen, Llama, Mistral, Gemma (2/3/4),
  Mixtral, LLaVA, Gemma-3 multimodal, Pixtral, Nomic, BGE, Jina,
  E5, DeepSeek-R1 and GPT-OSS families.

### Added — Modelfile renderer

- ``src/hfl/converter/modelfile.py`` compiles a ``ModelManifest`` back
  to a deterministic Modelfile string (``FROM`` / ``TEMPLATE`` /
  ``SYSTEM`` / ``PARAMETER`` / ``ADAPTER`` / ``LICENSE``). Byte-stable
  output so snapshot tests don't flake; handles stop-string escaping
  for quotes and backslashes. Consumed by ``/api/show`` and
  ``hfl show --modelfile``.

### Added — duration parser

- ``src/hfl/utils/duration.py`` implements Go-style duration parsing
  (subset that Ollama accepts): hour/minute/second/ms/us/µs/ns
  components, plain numbers as seconds, sentinels for "never expire"
  (-1) and "unload immediately" (0). Rejects booleans, "1d", "5minutes"
  and negative durations other than -1 — every one of those would be
  a silent behaviour-divergence from Ollama.

### Internal

- ``ServerState.keep_alive_deadline_for()`` and
  ``set_keep_alive_deadline()`` — per-model deadline storage keyed by
  name so hot-swaps preserve the field.
- All new routes wired via ``server.py`` with the existing auth +
  rate-limit + CORS + body-size middleware stack untouched.

### Metrics

| | 0.3.5 | 0.4.0 |
|-|-|-|
| Tests | 2137 | 2283 (+146) |
| Coverage | 89.01% | ~89% |
| Ollama REST endpoints covered | 4/16 | 10/16 |
| Ollama CLI commands covered | 5/11 | 8/11 |

Remaining Ollama-parity items (P0-1 embeddings, P0-5 structured
outputs, P0-6 vision, Modelfile ingestion, blobs, copy) are tracked
in ``OLLAMA_PARITY_PLAN.md`` and land in subsequent 0.4.x / 0.5.x
releases.

## [0.3.5] - 2026-04-17

Major internal overhaul closing the full backlog of the 2026-04-15
architecture / security / test-quality audit. Seven focused
refactors + two targeted test rounds; 24 new tests, 0 removed
regressions. Public behaviour is the same for well-formed clients;
three error envelopes changed shape — see "Breaking envelope
changes" below.

### Performance / reliability

- **``engine.unload()`` now runs off the event-loop thread**
  (``src/hfl/api/state.py``). ``set_llm_engine``, ``set_tts_engine``
  and ``cleanup`` wrap the synchronous teardown in
  ``asyncio.to_thread`` so ``/healthz``, ``/metrics`` and in-flight
  streams are no longer starved while a large model is being released
  (seconds on GPU / Metal). Guarded by four new concurrency tests in
  ``tests/test_state_concurrency.py::TestUnloadOffLoop`` — thread-id
  based, deterministic (30/30 runs), no timing thresholds.
- **``ModelPool.get_or_load`` polling loop replaced with
  ``asyncio.Event``** (``src/hfl/engine/model_pool.py``).
  ``_loading`` is now ``dict[str, asyncio.Event]`` instead of
  ``set[str]``; waiters ``await event.wait()`` with a 300 s cap and
  wake immediately when the owner's ``finally`` block ``set()``s the
  event. Old behaviour (3000 × 0.1 s polling) is gone; wake-up
  latency dropped from ~100 ms worst-case to one event-loop tick.
  ``TestModelPoolEventBasedWait`` pins the new contract (Event type,
  no ``asyncio.sleep(>=0.05)`` on the wait-path).

### Security

- **Cancel-safety stress test for the inference dispatcher**
  (``tests/test_dispatcher.py``). 200 random cancellations sweep
  every phase of ``InferenceDispatcher.slot()``; after the storm, a
  fresh ``slot()`` must succeed within 2 s, else a semaphore permit
  leaked. Confirms Python 3.10+ ``asyncio.Semaphore.acquire``
  cancel-safety holds.

### Configuration

- **5 magic-number timeouts lifted to ``HFLConfig``** with env-var
  overrides (``src/hfl/config.py``):
  - ``HFL_STREAM_QUEUE_PUT_TIMEOUT`` (default 60 s) — was hard-coded
    in ``async_wrapper.py``, ``vllm_engine.py``, ``streaming.py``.
  - ``HFL_STREAM_QUEUE_GET_TIMEOUT`` (default 30 s).
  - ``HFL_VLLM_ERROR_PUT_TIMEOUT`` (default 10 s).
  - ``HFL_VLLM_SHUTDOWN_JOIN_TIMEOUT`` (default 5 s).
  - ``HFL_REGISTRY_SQLITE_TIMEOUT`` (default 30 s).
- **HF_TOKEN platform-limitation note** added to
  ``src/hfl/config.py`` (immutable ``str`` → cannot be zeroed in
  memory; mitigations documented).

### Refactor / cleanup

- **28 ``HTTPException`` raises migrated to ``HFLError`` subclasses**
  in ``api/model_loader.py``, ``api/routes_tts.py``, and the
  validation paths of ``api/helpers.py``. Exception taxonomy:
  - ``ModelNotFoundError`` (new: ``status_code=404``).
  - ``ModelTypeMismatchError`` (new, carries ``model_name``,
    ``expected``, ``got``).
  - ``ModelNotReadyError`` (was already 503).
  - ``ValidationError`` (from ``hfl.exceptions.APIError`` — 400).
  - ``GenerationTimeoutError`` grows an optional ``operation`` label.
  The global handler now produces the standard envelope
  ``{"error": ..., "code": "<ExceptionClass>", "details": ...}`` for
  all of these, ending three ad-hoc 400-body shapes.
- **``prepare_stream_response`` helper** (``src/hfl/api/helpers.py``)
  extracts the ``acquire-slot + 429-pass-through + StreamingResponse``
  pattern duplicated in 5 streaming endpoints across OpenAI, Ollama
  and Anthropic routes.

### Tests

- **Targeted coverage** for previously uncovered error paths:
  - ``model_loader`` post-load cleanup (2 tests).
  - ``middleware._get_client_ip`` with invalid X-Forwarded-For (1).
  - ``routes_tts`` type-mismatch path (1).
  - ``model_pool`` polling retry / double-load race / lock-guarded
    cache hit / unload-failure logging (5).
  - ``prepare_stream_response`` happy + rejection paths (2).
  - CORS construction-time validator (4).
- **Tighter contracts** in ``tests/test_routes_health.py`` — 5
  ``*_returns_200`` tests upgraded to also assert body shape.
- **New pytest markers** ``slow`` and ``integration`` registered in
  ``pyproject.toml``; ``tests/stress/*`` classes annotated with
  ``@pytest.mark.slow``.

### Breaking envelope changes

Three JSON responses changed from ad-hoc dicts to the standard
envelope. If your client reads ``response.json()["detail"]``, add a
fallback to ``response.json().get("error")``:

- ``ModelNotFoundError`` (was ``{"detail": "Model not found: X"}``,
  now ``{"error": "Model not found: X", "code": "ModelNotFoundError",
  "details": "Use 'hfl list' ..."}``).
- ``ModelTypeMismatchError`` (was
  ``{"detail": {"code": "MODEL_TYPE_MISMATCH", "expected": ...,
  "got": ...}}``, now ``{"error": "Model 'X' is not a Y model",
  "code": "ModelTypeMismatchError", "details": "..."}``).
- ``ValidationError`` in options/request conversion (was
  ``{"detail": "..."}``, now standard envelope with
  ``code="ValidationError"``).

### Internal

- Python 3.16-ready: no remaining ``asyncio.iscoroutinefunction``
  callers (R1 fix in 0.3.3).

### Metrics

| | 0.3.4 | 0.3.5 |
|-|-|-|
| Tests | 2113 | 2137 |
| Coverage | 88.18% | 89.01% |
| `model_pool` branch coverage | 81% | 90% |
| `engine.unload` on event-loop thread | yes | no |
| Dispatcher cancel-safety test | no | yes (200 cycles) |
| Magic-number timeouts tunable | no | yes (5 env vars) |

## [0.3.4] - 2026-04-17

Tighter input validation and a misconfiguration guard for CORS. Closes
items from the second-pass security audit; no behaviour changes for
well-formed clients.

### Security

- **`seed` field now bounded to unsigned 32-bit** on OpenAI
  `ChatCompletionRequest` and `CompletionRequest` (`ge=0, le=2**32-1`).
  Previously unbounded; could pass integer-overflow values into
  backends that expect uint32.
- **`stop` sequences bounded** on OpenAI chat + completion: at most 10
  entries, each at most 256 characters. Prevents a client from forcing
  quadratic-per-token stop-sequence matching.
- **Anthropic `stop_sequences` bounded** (max 10 entries, 256 chars
  each via `field_validator`).
- **Anthropic `metadata` bounded**: at most 64 keys, key length
  ≤128 chars, string values ≤1024 chars. Prevents DoS via a
  multi-megabyte metadata dict.
- **TTS `voice` and `language` fields** now have `max_length` (128 and
  32 respectively) in both OpenAI and native TTS request schemas.
- **CORS misconfiguration rejected at construction**: `HFLConfig`
  raises `ValueError` when `cors_allow_credentials=True` is paired
  with wildcard origins (`cors_allow_all=True` or
  `cors_origins=["*"]`). Previously the combination was accepted
  silently even though every browser rejects it (W3C Fetch §3.2.1).

### Docs

- README.md / README.es.md version references updated from the stale
  "v0.3.0 alpha" text. The CORS-is-permissive bullet was also wrong
  (CORS defaults have been restrictive since 0.3.0); replaced with the
  current opt-in model.

### Tests

- Added boundary tests for `RequestBodyLimitMiddleware` (exact-at-limit
  accepted, one byte over rejected — guards against `>` vs `>=` drift).
- Added 4 tests for the new CORS validator.
- New pytest markers `slow` and `integration` registered in
  `pyproject.toml` so future filesystem / timing-dependent tests can
  be tagged and filtered on CI.

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
