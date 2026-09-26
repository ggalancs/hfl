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
| `HFL_QUEUE_MAX_INFLIGHT` / `HFL_NUM_PARALLEL` | `OLLAMA_NUM_PARALLEL`     | `1`     | Requests one model serves at once. Only backends that serve several at once use it (llama-server: default 4 when this is 1; vLLM); every other backend runs one request at a time, across all its models. |
| `HFL_QUEUE_MAX_SIZE` / `HFL_MAX_QUEUE`        | `OLLAMA_MAX_QUEUE`         | `16`    | Max wait queue; further requests get 429. |
| `HFL_QUEUE_ACQUIRE_TIMEOUT`    | —                          | `60`    | Seconds a caller may wait for a slot before 503. |
| `HFL_MAX_LOADED_MODELS`        | `OLLAMA_MAX_LOADED_MODELS` | `0`     | Optional ceiling on the **number** of resident models. `0` (default) = no ceiling: memory alone decides, see `HFL_MEMORY_BUDGET`. When set, loading one more model than this unloads the least recently used idle one. |
| `HFL_MEMORY_BUDGET`            | —                          | `85`    | Share of **total RAM** the machine may have in use after a model loads (other programs included), as a percentage (`85` or `85%`). HFL keeps as many models loaded as fit under it: a load that does not fit unloads idle models, least recently used first; a model a request is using is never unloaded — the load waits for it (up to `HFL_QUEUE_ACQUIRE_TIMEOUT`, then 503); a model that cannot fit even alone is refused (507) with the numbers, before anything is unloaded. `/api/ps` reports the budget and what is in use. With an NVIDIA GPU (read through `nvidia-smi`), a model must also fit the card's memory under the same percentage; a GPU whose memory cannot be read (ROCm, CUDA without `nvidia-smi`) keeps one model loaded at a time unless `HFL_MAX_LOADED_MODELS` says otherwise. `HFL_DISABLE_MEMORY_PREFLIGHT=1` turns the memory checks off (only the count ceiling remains). |
| `HFL_DISABLE_MEMORY_PREFLIGHT` | —                          | unset   | `1` skips every memory check: the per-load llama.cpp preflight and the residency budget. For hosts whose real limit is a discrete GPU's VRAM, which these checks do not measure. |

> **Parallel requests need a backend that batches them: llama-server or vLLM.**
> The default GGUF backend (llama-cpp-python) and Transformers drive a single
> non-reentrant model instance with one KV cache: two overlapping requests would
> interleave their state and produce corrupted text, not an error, so they share
> one queue, clamped to 1 slot (a warning is logged). llama-server and vLLM get a
> queue of their own per model, with `HFL_NUM_PARALLEL` slots (llama-server: 4
> when it is left at 1) — their requests neither wait behind each other nor
> behind another model's.

## Lifecycle / keep-alive

| HFL                | Ollama alias        | Default | What it does |
|--------------------|---------------------|---------|--------------|
| `HFL_KEEP_ALIVE`   | `OLLAMA_KEEP_ALIVE` | `5m`    | How long a model stays loaded after its last use; enforced — a model idle past it, and not in use, is unloaded (checked every 15 s). The clock restarts on every request and when the request ends, so a model in continuous use never expires between turns. A `keep_alive` sent on a request is remembered for that model and wins over this default; `0` unloads after that one response; `-1` keeps it until `hfl stop` or a load needs the memory. Ollama duration grammar (`5m`, `30s`, `0`, `-1`); an unreadable value falls back to `5m`. |

## Backend selection / runtime

| HFL                  | Ollama alias            | Default | What it does |
|----------------------|-------------------------|---------|--------------|
| `HFL_LLM_LIBRARY`    | `OLLAMA_LLM_LIBRARY`    | (auto)  | Pin auto-selection to a specific backend: `llama-cpp`, `llama-server`, `transformers`, `vllm`, `mlx`. `llama-server` serves GGUF models only (others keep their backend) and needs llama.cpp's `llama-server` installed. Per-call `backend=` argument still wins. |
| `HFL_LLAMA_SERVER_BIN` | — | (PATH) | The `llama-server` executable to run, when it is not on the PATH. |

> On the command line: `hfl serve --parallel N` asks for N requests at once per
> GGUF text model (it implies `llama-server`), and `hfl serve --backend NAME`
> picks a backend. The choice stays per model either way: a vision GGUF (an
> `mmproj-*.gguf` beside it) and non-GGUF models keep their usual backend.
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

## Compliance / pull governance

`pull`, `push` and smart-pull are **owner** (administrative) operations:
they download or upload arbitrary repos on the server host. Over the
network these are refused for remote callers so an API *user* cannot
provision models or "accept" licenses on the owner's behalf. The
interactive CLI (`hfl pull`) is unaffected — it always prompts a human.

| HFL                     | Ollama alias | Default        | What it does |
|-------------------------|--------------|----------------|--------------|
| `HFL_ALLOW_REMOTE_PULL` | —            | `false`        | When truthy, allows non-loopback callers to hit `/api/pull`, `/api/pull/smart` and `/api/push`. Default refuses them with `403 remote_admin_forbidden`. Enable only if you knowingly administer this server remotely (the API key still guards it). |
| `HFL_GENERATION_TIMEOUT` | —           | `600`          | Seconds a single inference may run before the server answers 504. Generous for chat, short for long-form generation: a 70B at ~7 tok/s needs ~5 min for 2000 tokens *plus* prompt processing, so raise it if you set a high `num_predict`. Your HTTP client's timeout should be larger than this, so the server cuts first and you get a clean 504. |
| `HFL_MODEL_LOAD_TIMEOUT` | `OLLAMA_LOAD_TIMEOUT` | `300` | Seconds a cold model load may take. A 44 GiB GGUF that is not in the page cache can exceed this on slow storage. |
| `HFL_ALLOW_AGENT_LOOP`  | —            | `false`        | When truthy, allows `/api/chat` requests to set `agent_loop: true`, which makes the server dispatch MCP tool calls on the caller's behalf. The operator chooses which MCP servers are connected, but the request supplies the prompt that steers *which* tool runs with *which* arguments — so with capable tools connected this hands their reach to anyone who can reach the API. Default refuses with `403 agent_loop_disabled`. |
| `HFL_METRICS_PUBLIC`    | —            | `false`        | When truthy, serves `/metrics` and `/metrics/json` without the API key. They expose request and token volumes, per-endpoint counters and live queue depth — a usage side-channel. Enable only for a Prometheus that cannot authenticate. Has no effect when no API key is configured (nothing is authenticated then). |
| `HFL_ACCEPT_NETWORK_EXPOSURE` | —      | `false`        | Unattended consent for binding a non-loopback address. `hfl serve` warns and asks for confirmation before exposing the API; without a TTY (systemd, launchd) there is nobody to ask, so it refuses unless this is truthy, an API key is set (the exposure is authenticated), or it runs in a container (Docker, Podman, Kubernetes), where the bind reaches only as far as the published ports — it then says so and recommends a key. |
| `HFL_API_KEY`           | —            | —              | The key `hfl serve` requires on every request (`Authorization: Bearer <key>` or `X-API-Key`), as `--api-key` does — better than the flag, which every local user can read in `ps`. |
| `HFL_LICENSE_POLICY`    | —            | `permissive`   | Which license risk tiers the owner pre-accepts for non-interactive (HTTP API) pulls. Cumulative: `permissive` (Apache/MIT/BSD-class only) → `conditional` (also Llama/Gemma/Qwen/OpenRAIL) → `all` (also non-commercial / restricted / unknown). A license outside the tier is refused with a `license_not_accepted` event; widen the policy or pull it locally via the CLI. |

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
| `HFL_MLX_PROMPT_CACHE_BYTES`      | `2147483648` | Bytes of KV cache the MLX backend keeps for recent prompts, so a follow-up chat turn only evaluates what is new. Shares unified memory with the model weights, hence the cap. `0` disables it. On a server shared by several people, note that any prompt cache (this one, and llama.cpp's reuse of the previous prompt) makes a request faster when its prefix was sent recently — by anyone — and the response's timing fields show it; set `0` if users must not be able to probe each other's prompts that way. |
| `HFL_VLLM_ERROR_PUT_TIMEOUT`       | `10`    | Shorter window for the vLLM error sentinel. |
| `HFL_VLLM_SHUTDOWN_JOIN_TIMEOUT`   | `5`     | vLLM worker join timeout on shutdown. |

## Storage / registry

| HFL                            | Default | What it does |
|--------------------------------|---------|--------------|
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
