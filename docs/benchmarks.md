# Benchmarks: HFL, Ollama and llama-server

One GGUF file, three servers, the same requests. Reproduce it with:

```bash
pip install "hfl[llama]" && brew install ollama llama.cpp     # or your platform's packages
python scripts/bench_compare.py hf.co/bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M
```

## Result — 2026-09-24

Apple M3 Max, 128 GB, macOS 26.6.2, on AC power.
[Phi-3.5-mini-instruct Q4_K_M](https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF) (MIT, 2.2 GB).
HFL 0.21.0 (llama-cpp-python 0.3.34), Ollama 0.34.2, llama-server build 10964.
Medians of 5 interleaved rounds. [Raw data](benchmarks/2026-09-24-m3-max-phi-3.5-mini.json).

| | HFL | Ollama | llama-server |
|---|---:|---:|---:|
| Time to first token, short prompt | 0.094 s | 0.116 s | 0.162 s |
| Time to first token, ~2,000-token prompt | 3.27 s | 3.43 s | 3.21 s |
| Decode, one request | 56.6 tok/s | 51.4 tok/s | 56.5 tok/s |
| Throughput, 4 requests at once | 59.3 tok/s | 51.5 tok/s | **83.1 tok/s** |

What it says:

- **One request at a time, the three are close.** HFL and llama-server were
  level; Ollama decoded about 10% slower.
- **Several requests at once, llama-server is ahead** — it decodes them in
  parallel slots of one batch. HFL runs one inference at a time (by design:
  one shared model instance), so four requests take about four times as long
  as one. Ollama's throughput with four requests matched its single-request
  rate: with its default settings it also served them one after another.

Three sessions were run that evening (the first two measuring the servers one
after another, the last interleaved). In all three, llama-server led with four
requests at once and HFL came second, and HFL's single-request decode was the
highest or level with llama-server's. The other differences — time to first
token on the short prompt, for one — changed order between sessions and are
within the noise. The absolute numbers moved much more: the machine was a
desktop in use (browser, editor, the window server sharing the GPU), and every
server ran 30–40% slower in the last session than in the first. That is why
the script interleaves the servers in rotating rounds — a drift then hits all
three alike — and why a comparison is only meaningful between servers measured
together, never across sessions.

## Method

- **Same file.** `hfl pull` fetches the GGUF once; HFL serves it by name,
  llama-server by path (`-m`), Ollama through a Modelfile `FROM` it, created in
  a throwaway `OLLAMA_MODELS` (your `~/.ollama` is not touched).
- **Same settings.** Context 4096 for all (`hfl serve --ctx`, `llama-server -c`,
  Ollama's `PARAMETER num_ctx`), full GPU offload, temperature 0, seed 42,
  256 tokens at most. Everything else is each server's default.
- **Same API.** All three are measured through their OpenAI-compatible
  `/v1/chat/completions`, streaming, from the client.
- **No cache hits.** Every request starts with a unique line, so no server can
  answer from a cached prompt. (The first version of the script repeated the
  prompt and measured 0.026 s to first token on a 2,000-token prompt: prefix
  caching, not prefill.)
- **Tokens counted by the script**, with the model's own tokenizer, the same
  way for all three — not from each server's `usage` or its chunking. All
  three produced the same number of tokens for the same prompt.
- **Decode rate** = tokens after the first ÷ time between the first and the
  last content chunk. **Throughput** = all tokens of the 4 simultaneous
  requests ÷ wall time.
- One warm-up request per server before measuring; servers are started and
  stopped by the script.

## Before you trust a number

- Check the power source: on a laptop, battery can cut decode speed several
  times.
- Close what you can; the GPU is shared with the display and the browser.
- One model on one machine. A different size, quantization or backend (MLX,
  CUDA) can order them differently — run the script on yours.
