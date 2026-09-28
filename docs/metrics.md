# Metrics

`GET /metrics` serves Prometheus text; `GET /metrics/json` a JSON summary.
With an API key set, both need it (`HFL_METRICS_PUBLIC=true` serves them
without it: see [env-vars.md](env-vars.md)).

```yaml
# prometheus.yml
scrape_configs:
  - job_name: hfl
    static_configs:
      - targets: ["127.0.0.1:11434"]
    # with an API key:
    # authorization: { credentials: "<HFL_API_KEY>" }
```

## Generation

| metric | type | what it measures |
|---|---|---|
| `hfl_generation_tokens_per_second` | histogram | Speed of generation, over the decode phase only: prompt processing runs about ten times faster, and tokens ÷ total time mixes the two into a number that is neither. Non-streamed replies use the engine's own timing; streams, the time from the first token to the last. |
| `hfl_time_to_first_token_ms` | histogram | Streamed replies: from the start of the reply to its first token. The model load, if the request caused one, is not included. |
| `hfl_generation_latency_ms` | histogram | Whole generation, prompt processing included. |
| `hfl_tokens_generated_total`, `hfl_tokens_input_total` | counter | Tokens out and in, streams included. |

## Models and memory

| metric | type | what it measures |
|---|---|---|
| `hfl_models_loaded` | gauge | Language models resident now. |
| `hfl_models_memory_bytes` | gauge | Estimated memory of those models (weights + KV cache). |
| `hfl_memory_total_bytes`, `hfl_memory_in_use_bytes` | gauge | The machine's memory, and how much is in use (other programs included). |
| `hfl_memory_budget_bytes` | gauge | What may be in use after a load (`HFL_MEMORY_BUDGET`). |
| `hfl_gpu_memory_total_bytes`, `hfl_gpu_memory_in_use_bytes` | gauge | An NVIDIA GPU's memory, when `nvidia-smi` can read it. |
| `hfl_model_loads_total`, `hfl_model_unloads_total` | counter | Loads and unloads. |

The memory figures describe the machine, so they go to a local caller only
(a Prometheus on the same host), as in `/api/ps`; a remote scraper gets the
rest.

## Queue and requests

| metric | type | what it measures |
|---|---|---|
| `hfl_inference_queue_depth` | gauge | Requests waiting for an inference slot. |
| `hfl_inference_concurrency_inflight`, `hfl_inference_concurrency_max` | gauge | Requests generating now, and the slots (`HFL_NUM_PARALLEL`). |
| `hfl_inference_accepted_total` | counter | Requests admitted to the queue. |
| `hfl_inference_rejected_full_total` | counter | Refused with 429: the queue was full. |
| `hfl_inference_rejected_timeout_total` | counter | Refused with 503: waited too long for a slot. |
| `hfl_requests_total`, `hfl_requests_by_endpoint_total`, `hfl_requests_by_status_total` | counter | HTTP requests. |
| `hfl_request_latency_ms` | histogram | HTTP request latency. |
| `hfl_errors_total` | counter | Errors, by type. |
| `hfl_ws_cancels_total`, `hfl_ws_cancel_orphans_total`, `hfl_stream_cancel_orphans_total` | counter | Cancelled streams, and those whose engine call kept running after the client left. |
| `hfl_uptime_seconds` | gauge | Time since the server started. |
