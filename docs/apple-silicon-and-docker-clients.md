# Running HFL on Apple Silicon behind a containerized app

Guidance for the common setup where an application runs in Docker and calls
HFL over the Ollama-compatible API. Everything below was measured on a
MacBook Pro M3 Max (40-core GPU, 128 GB unified memory) serving
Qwen2-72B Q4_K_M and Qwen3-14B Q4_K_M.

The headline: **HFL detects and uses the GPU automatically — but two
environment factors can cost you an order of magnitude, and neither is
visible from inside the application.**

## 1. Never run HFL itself inside a container on macOS

Docker's own documentation is explicit: *"GPU support in Docker Desktop is
only available on Windows with the WSL2 backend."* On macOS, containers run
inside a Linux VM with **no access to the Apple GPU**. HFL inside that
container cannot use Metal at all — it falls back to CPU, which on a 72B
model is the difference between usable and unusable.

The correct topology, and the one HFL is designed for:

```
┌─ macOS host ────────────────────────────────────────────┐
│                                                          │
│   hfl serve --port 11434    ← Metal, full GPU access     │
│           ▲                                              │
│           │ http://host.docker.internal:11434            │
│   ┌───────┴──────────────────────────────────────────┐   │
│   │ Docker VM: app containers, databases, workers    │   │
│   └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

The `hfl` Docker images shipped by this project are for Linux hosts with
NVIDIA GPUs, or for CPU-only serving. Do not use them as the inference
backend on a Mac.

### Reaching the host from a container

HFL binds to `127.0.0.1` by default, which a container cannot reach. Either:

- point clients at `http://host.docker.internal:11434` **and** start the
  server with `HFL_HOST=0.0.0.0` (or `OLLAMA_HOST=0.0.0.0:11434`), or
- keep the default bind and use `--network host` (Linux only).

Exposing `0.0.0.0` puts the inference API on your LAN. Set `HFL_API_KEY` if
the machine is not on a trusted network.

## 2. Plug the laptop in

This is the single largest lever, and it is invisible from the application:
every layer still reports as offloaded to Metal, the model just crawls.

Qwen3-14B Q4_K_M, identical build, identical prompt:

| Power source | Prefill    | Generation  |
| ------------ | ---------- | ----------- |
| Battery      | 113 tok/s  | 4.7 tok/s   |
| AC power     | 393 tok/s  | 38.7 tok/s  |

**8× on token generation.** macOS clocks the GPU down aggressively on
battery; the commonly quoted "30-50% penalty" understates it badly for
sustained inference. On the 72B the same switch took generation from
3.5 tok/s to 9.0 tok/s.

Since 0.17.0 HFL warns about this on every load, and `hfl doctor` reports
the current power source.

## 3. Give the Docker VM only what it needs

On Apple Silicon the CPU and GPU share one memory pool *and* one power
budget. A VM allocated every core competes with Metal for both. Docker
Desktop → Settings → Resources:

| Setting | Typical default | Suggested | Why |
| --- | --- | --- | --- |
| CPUs | all cores (16) | 4-6 | Leaves the P-cores and the power budget to the GPU. App containers and databases are I/O-bound, not CPU-bound. |
| Memory | 8 GiB | what the containers actually use + ~1 GiB | Every GiB reserved by the VM is a GiB unavailable for model weights and KV cache. The host process shows the *full* allocation as resident. |
| Swap | 1 GiB | 1 GiB | Fine as-is. |

Measure before you cut, rather than guessing:

```bash
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

In the measured setup, nine containers totalled ~1.7 GiB of a 7.65 GiB
allocation, with a single Neo4j instance accounting for 1.39 GiB of it —
so ~6 GiB of host RAM was reserved and never used.

Also worth enabling: **Resource Saver** (idles the VM when no containers are
running) and **VirtioFS** for file sharing.

### Tune the memory hog, not the VM

Java-based services size their heap from the *container's* view of memory,
so shrinking the VM without configuring them just moves the problem. For
Neo4j:

```yaml
environment:
  NEO4J_server_memory_heap_initial__size: 512m
  NEO4J_server_memory_heap_max__size: 1g
  NEO4J_server_memory_pagecache__size: 512m
```

Postgres and Redis are usually already small; check before touching them.

### Do not bind-mount the model directory

Mounting `~/.hfl/models` into a container routes multi-GB reads through the
VM's filesystem layer for no benefit — the container is not doing inference.

## 4. Client-side settings that matter

For applications talking to HFL over the Ollama API:

**Use one `num_ctx` across every service.** Since 0.17.0, `options.num_ctx`
is honoured, and a value that differs from the resident engine's context
triggers a **full model reload** — tens of seconds, and 44 GiB re-read, for a
72B. Three microservices requesting three different contexts will thrash the
model back and forth. Pick one value and set it everywhere.

**Size `num_ctx` deliberately.** The KV cache is `2 × n_layers × n_kv_heads ×
head_dim × 2 bytes` per token. On a 72B that is ~0.31 MiB/token: 16384 tokens
costs 5 GiB on top of 44 GiB of weights. `HFL_KV_CACHE_TYPE=q8_0` halves that
at negligible quality cost.

**Set `keep_alive` generously.** `"30m"` or `"-1"` (never expire) avoids
paying the load cost repeatedly. Do not send `keep_alive: 0` from a service
that will be called again shortly.

**Expect queueing, and honour it.** HFL serializes inference — llama.cpp
holds one non-reentrant model instance, so concurrent requests would corrupt
the KV cache. A full queue returns **429 with `Retry-After`**; an acquire
timeout returns **503**. Clients should retry on both rather than treating
them as hard failures. Live depth is in the `X-Queue-Depth` header and
`GET /healthz`.

**Set generous client timeouts.** A cold load of a 70B plus generation can
exceed default HTTP timeouts. The first request after a restart pays the
full load.

## 5. Verifying acceleration is actually on

Since 0.17.0 the answer is one line in the server log at INFO:

```
Acceleration: MTL0 (Apple M3 Max) · 81/81 layers on GPU · 44.2 GiB in MTL0 buffers
```

Over the API, `GET /api/ps` carries the same information per loaded model:

```json
{"models": [{"name": "…", "details": {
  "acceleration": "MTL0 (Apple M3 Max) · 81/81 layers on GPU · 44.2 GiB in MTL0 buffers",
  "context_size": 16384
}}]}
```

And `hfl doctor` reports backends, accelerators and power source without
loading anything.

If `acceleration` is absent or reports CPU-only, check in this order:
`hfl doctor` (is `gpu_offload=✓`?), whether HFL is running inside a
container, and `HFL_DISABLE_MEMORY_PREFLIGHT` / `n_gpu_layers` overrides.

## 6. Rough expectations on an M3 Max (plugged in)

| Model | Quant | Generation |
| --- | --- | --- |
| 14B | Q4_K_M | ~38 tok/s |
| 72B | Q4_K_M | ~9 tok/s |

Token generation is memory-bandwidth bound: divide the machine's bandwidth
(400 GB/s on the 40-core M3 Max) by the model's on-disk size for a ceiling.
Prefill is compute bound and much faster. If you are far below these numbers
with every layer offloaded, check power source first, then contention from
other processes.
