# HFL environment variables

Reference for every env var the HFL server reads at boot. Each row
lists the **HFL variable**, its **Ollama-equivalent** (when one
exists), the **default**, and what it controls. Variables are read
once at process start unless noted otherwise.

Resolution rule across the table: HFL-specific name wins, then HFL
alias, then Ollama-equivalent. So a host that already has
`OLLAMA_HOST` set works as a drop-in replacement; a host with both
`HFL_HOST` and `OLLAMA_HOST` honours `HFL_HOST`.

## Server bind

| HFL                     | Ollama alias    | Default       | What it does |
|-------------------------|-----------------|---------------|--------------|
| `HFL_HOST`              | `OLLAMA_HOST`*  | `127.0.0.1`   | Interface to bind. `OLLAMA_HOST` accepts `host`, `host:port`, or `:port`. |
| `HFL_PORT`              | `OLLAMA_PORT`*  | `11434`       | TCP port. Falls back to the port part of `OLLAMA_HOST` when set. |
| `HFL_HOME`              | —               | `~/.hfl`      | Root directory for models, registry, blobs. |

\* `OLLAMA_HOST` may carry the port; `OLLAMA_PORT` is consulted when
the host string does not.

## Concurrency / queue

| HFL                            | Ollama alias               | Default | What it does |
|--------------------------------|----------------------------|---------|--------------|
| `HFL_QUEUE_MAX_INFLIGHT` / `HFL_NUM_PARALLEL` | `OLLAMA_NUM_PARALLEL`     | `1`     | Inference slots executing simultaneously per model. |
| `HFL_QUEUE_MAX_SIZE` / `HFL_MAX_QUEUE`        | `OLLAMA_MAX_QUEUE`         | `16`    | Max wait queue; further requests get 429. |
| `HFL_QUEUE_ACQUIRE_TIMEOUT`    | —                          | `60`    | Seconds a caller may wait for a slot before 503. |
| `HFL_MAX_LOADED_MODELS`        | `OLLAMA_MAX_LOADED_MODELS` | `1`     | Models kept resident in `ModelPool` (LRU eviction). |

## Lifecycle / keep-alive

| HFL                | Ollama alias        | Default | What it does |
|--------------------|---------------------|---------|--------------|
| `HFL_KEEP_ALIVE`   | `OLLAMA_KEEP_ALIVE` | `5m`    | Default keep-alive applied when a request omits the field. Per-request value always wins; a previously-recorded deadline is preserved. Accepts the Ollama duration grammar (`5m`, `30s`, `0`, `-1`). |

## Backend selection / runtime

| HFL                  | Ollama alias            | Default | What it does |
|----------------------|-------------------------|---------|--------------|
| `HFL_LLM_LIBRARY`    | `OLLAMA_LLM_LIBRARY`    | (auto)  | Pin auto-selection to a specific backend: `llama-cpp`, `transformers`, `vllm`, `mlx`. Per-call `backend=` argument still wins. |
| `HFL_DISABLE_MLX`    | —                       | `0`     | When truthy, disables the MLX path on Apple Silicon (forces llama-cpp Metal). Useful for benchmarking. |
| `HFL_KV_CACHE_TYPE`  | `OLLAMA_KV_CACHE_TYPE`  | `f16`   | KV cache dtype: `f16`, `q8_0`, `q4_0`. Halves / quarters VRAM at the cost of accuracy. |
| `HFL_FLASH_ATTENTION`| `OLLAMA_FLASH_ATTENTION`| (auto)  | Toggle flash-attention fleet-wide (`1`/`0`). Per-load kwarg wins; per-arch safety list still rejects known-unsafe arches. |
| `HFL_DEFAULT_CTX_SIZE` | —                     | `0`     | Default `n_ctx`. `0` = auto-detect from GGUF metadata. |

## Security / CORS / Rate-limit

| HFL                          | Ollama alias       | Default                  | What it does |
|------------------------------|--------------------|--------------------------|--------------|
| `HFL_ORIGINS`                | `OLLAMA_ORIGINS`   | (same-origin)            | Comma-separated CORS allow-list. `*` flips wildcard mode (and rejects credentials). |
| `HFL_RATE_LIMIT_ENABLED`     | —                  | `true`                   | Master switch for the in-process rate limiter. |
| `HFL_RATE_LIMIT_REQUESTS`    | —                  | `60`                     | Requests per window. |
| `HFL_RATE_LIMIT_WINDOW`      | —                  | `60`                     | Window size in seconds. |
| `HFL_MAX_REQUEST_BYTES`      | —                  | `10485760` (10 MiB)      | Cap on request body. `0` disables. |

## Observability

| HFL                  | Ollama alias       | Default | What it does |
|----------------------|--------------------|---------|--------------|
| `HFL_DEBUG`          | `OLLAMA_DEBUG`     | (off)   | Truthy values force the `hfl` root logger to DEBUG. |
| `HFL_AUDIT_LOG_PATH` | —                  | (off)   | When set, audit events are appended to that file (with rotation). |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | —         | (off)   | Standard OpenTelemetry env. HFL emits spans when this is set. |

## Streaming backpressure

| HFL                                | Default | What it does |
|------------------------------------|---------|--------------|
| `HFL_STREAM_QUEUE_PUT_TIMEOUT`     | `60`    | Seconds the engine thread will wait to enqueue a token before raising. |
| `HFL_STREAM_QUEUE_GET_TIMEOUT`     | `30`    | Seconds the consumer waits for the next token. |
| `HFL_VLLM_ERROR_PUT_TIMEOUT`       | `10`    | Shorter window for the vLLM error sentinel. |
| `HFL_VLLM_SHUTDOWN_JOIN_TIMEOUT`   | `5`     | vLLM worker join timeout on shutdown. |

## Storage / registry

| HFL                            | Default | What it does |
|--------------------------------|---------|--------------|
| `HFL_REGISTRY_SQLITE_TIMEOUT`  | `30`    | SQLite busy-timeout in seconds before raising `OperationalError`. |
| `HF_TOKEN`                     | (none)  | Standard HuggingFace token. Read once at boot, held in memory only. |

## Notes

- Variables not listed here may exist in the codebase but are
  considered internal — they are not part of the documented operator
  surface and may change without a deprecation period.
- A truthy value is one of `1`, `true`, `yes`, `on`
  (case-insensitive). Anything else is treated as falsy.
- An invalid value (e.g. a malformed `OLLAMA_HOST`, an unknown
  `HFL_LLM_LIBRARY` backend) is logged once and ignored — server boot
  must not fail because of operator misconfiguration.
