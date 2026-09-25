<div align="center">

# HFL

**Download, run and try any Hugging Face model on your own machine.**

One command takes a model from the Hub to a local chat or to an OpenAI-,
Ollama- and Anthropic-compatible API. No account. No cloud of its own — by design.

[![PyPI](https://img.shields.io/pypi/v/hfl.svg)](https://pypi.org/project/hfl/)
[![Python](https://img.shields.io/pypi/pyversions/hfl.svg)](https://pypi.org/project/hfl/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/ggalancs/hfl/blob/main/LICENSE)
[![Docker](https://img.shields.io/badge/docker-ghcr.io%2Fggalancs%2Fhfl-2496ED.svg)](https://github.com/ggalancs/hfl/pkgs/container/hfl)
[![CI](https://github.com/ggalancs/hfl/actions/workflows/ci.yml/badge.svg)](https://github.com/ggalancs/hfl/actions/workflows/ci.yml)

[Quick start](#quick-start) · [Why HFL](#why-hfl) · [Install](#install) · [Use it](#use-it) · [API](#connect-your-tools) · [Docs](#documentation) · **[Español](https://github.com/ggalancs/hfl/blob/main/README.es.md)**

</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/ggalancs/hfl/main/docs/assets/hfl-run-demo.svg" alt="Terminal: hfl run downloads a model from the Hugging Face Hub on first use and starts a local chat (real output, abridged)" width="880">
</p>

## Quick start

```bash
pip install "hfl[llama,mlx]"        # the MLX part installs only on Apple Silicon

hfl run hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
```

The model is downloaded the first time and reused from disk after that. To serve it instead:

```bash
hfl serve --model hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
```

Any OpenAI, Ollama or Anthropic client can now talk to `http://localhost:11434` — and
opening that address in a browser gives you a chat page, served by HFL itself
(no account, nothing loaded from the internet).

## Why HFL

- **The whole Hub, not one file format.** GGUF repos run through llama.cpp, MLX
  builds run natively on Apple Silicon, and safetensors checkpoints are converted
  and quantized for you on pull. Copy a repo name from the Hub and run it.
- **Browse the Hub from your terminal.** `hfl search` pages through the live
  Hub with sizes, downloads and formats at a glance; filter by GGUF or by size,
  press a number and the model is pulled, license check included.
- **Yours alone.** No account, no sign-in, no cloud service behind it. On its own
  HFL talks to one server, the Hugging Face Hub, to fetch weights; its web-search
  endpoints reach the web only when a client calls them, and a test pins every
  host the code can reach. With no network, everything you have already pulled
  keeps working.
- **Plugs into what you already use.** OpenAI (chat, completions, embeddings,
  Responses), Ollama and Anthropic Messages APIs on one port, with structured
  tool calling for Qwen, Llama 3, Mistral, Gemma 4, gpt-oss, DeepSeek, GLM and
  Hermes families and JSON-schema outputs.
  Existing `OLLAMA_*` settings such as `OLLAMA_HOST` are honoured.
- **As many models as your memory holds.** Before every load HFL estimates what
  the model will take (weights + KV cache) and keeps the machine under a memory
  budget you set: models load side by side while they fit, idle ones make room,
  one in use is never pulled out from under a request, and one that cannot fit
  is refused with the numbers — before anything is unloaded. GPU-aware on NVIDIA.
- **Knows the Hub.** Find models that fit your hardware (`hfl recommend`), pick
  the best community quant for your machine (`hfl pull-smart`), size
  Mixture-of-Experts models by their total parameters, check licenses before
  downloading and keep a provenance record of every pull.

## Install

| How | Command |
|---|---|
| **pip** (recommended) | `pip install "hfl[llama,mlx]"` |
| **Docker** | `docker run -p 11434:11434 -v hfl:/var/lib/hfl ghcr.io/ggalancs/hfl` |
| **Installers** | `.dmg`, `.msi` and standalone binaries on the [releases page](https://github.com/ggalancs/hfl/releases) |
| **From source** | `git clone https://github.com/ggalancs/hfl && cd hfl && pip install -e ".[llama,mlx]"` |

<details>
<summary><b>Optional extras</b> — GPU, speech, vLLM and more</summary>

| Extra | Adds |
|---|---|
| `llama` | llama.cpp for GGUF models (Metal on Apple Silicon out of the box) |
| `mlx` | Native MLX backend on Apple Silicon |
| `transformers` | Transformers backend for GPU inference with bitsandbytes |
| `vllm` | vLLM backend |
| `convert` | Tools to convert safetensors to GGUF |
| `tts` / `coqui` | Text-to-speech (Bark, SpeechT5 / Coqui XTTS, VITS) |
| `stt` | Speech-to-text (Whisper) |
| `mcp` | Model Context Protocol client and server |
| `all` | Everything above |

Converting safetensors to GGUF builds llama.cpp's tools the first time, which
needs **git**, **cmake** and a **C++ compiler** (`xcode-select --install` on
macOS, `sudo apt install build-essential cmake` on Debian/Ubuntu). Pre-quantized
GGUF and MLX models need none of this.

</details>

## Use it

### Search and pick from the terminal

<p align="center">
  <img src="https://raw.githubusercontent.com/ggalancs/hfl/main/docs/assets/hfl-search-demo.svg" alt="Terminal: hfl search lists Hugging Face Hub models page by page with size, downloads and format; pressing a number pulls the model (real output)" width="880">
</p>

```bash
hfl search qwen3                          # everything matching, most downloaded first
hfl search qwen3 --gguf --max-params 8    # GGUF only, 8B or smaller — searched across the whole Hub
hfl search llama --sort likes             # or: downloads (default), created
```

Each page lists up to ten models with their size, downloads, likes, format and
task. Press **0–9** to pull one (you confirm, and its license is checked
first), **SPACE** for the next page, **p** for the previous one, **q** to leave.
Sizes are total parameters, so Mixture-of-Experts models are not shown smaller
than they are.

### Chat

```bash
hfl run hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M   # from the Hub, pulled on first use
hfl run qwen3-coder                                          # a short name, looked up on the Hub
hfl run hf.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit       # an MLX build on Apple Silicon
hfl run llama70b --system "You are a Python expert"          # a local name or alias
hfl run llama70b --session work                              # resume and save a conversation
```

The `hf.co/` prefix is optional. `:Q4_K_M` picks a quantization, `@<ref>` pins a
branch, tag or commit.

A short name (`qwen3-coder`, `llama3.2`, `qwen3:8b`, `gemma3:4b-q8_0`) that is
not a local model is looked up among the Hub's GGUF builds: instruct builds and
the usual quantizers first, derivatives (abliterated, merges...) left out, and a
quantization that fits your machine (Q4_K_M unless it does not). You pick from the
list before anything is downloaded (`--yes` takes the first), and the name is kept
as an alias, so the next `hfl run qwen3-coder` is local. Ollama clients get the
same over `/api/pull {"model": "llama3.2"}`.

### Pull, search and manage

```bash
hfl pull meta-llama/Llama-3.3-70B-Instruct                 # Q4_K_M by default
hfl pull meta-llama/Llama-3.3-70B-Instruct --quantize Q5_K_M --alias llama70b
hfl pull meta-llama/Llama-3.3-70B-Instruct@a1b2c3d          # reproducible: pinned revision

hfl list                                # what is on this machine
hfl inspect llama70b                    # details and license
hfl rm llama70b
```

### Find the right model

```bash
hfl recommend                           # top models that fit THIS machine's RAM/VRAM
hfl discover --family qwen              # filter the live Hub; marks what you already have
hfl pull-smart Qwen/Qwen3-30B-A3B       # best community variant for your hardware
hfl verify <model>                      # tokenizer, chat template, smoke generation, tools
hfl bench <model>                       # time to first token, tokens/s, p50/p95
```

See [docs/hub-native-features.md](https://github.com/ggalancs/hfl/blob/main/docs/hub-native-features.md) for every option.

### Several models at once

HFL keeps every model that fits under `HFL_MEMORY_BUDGET` — the share of total
RAM the machine may have in use after a load, other programs included (default
`85%`). Each load reports the numbers before it happens:

```text
Memory: 65.3 of 128.0 GB in use (51%). qwen3-14b needs ~9.0 GB → after loading, 74.3 GB in use (58%); budget 85%.
```

- fits → it loads next to the models already resident;
- does not fit → idle models are unloaded, least recently used first;
- the room is held by models answering requests → the load waits for them;
- cannot fit even alone → refused with the numbers (HTTP 507), nothing unloaded.

With an NVIDIA GPU the model must also fit the card (read through `nvidia-smi`).
Idle models unload after `keep_alive` (default `5m`, renewed on every use).
`hfl ps` and `GET /api/ps` show what is loaded and how much room is left.

<details>
<summary><b>Tool calling</b> — agents work out of the box</summary>

Send `tools` on `/api/chat`, `/v1/chat/completions` or `/v1/messages`: HFL
renders them through the model's own chat template (Qwen and Hermes
`<tool_call>`, Llama 3 `<|python_tag|>`, Mistral `[TOOL_CALLS]`, gpt-oss's
Harmony channels, DeepSeek's and GLM's own markers), parses the reply into
`message.tool_calls` with the arguments as an object, and accepts
`role: "tool"` results on the next turn. When a model's template has no place
for tools (Hermes-3, DeepSeek-R1), HFL writes them into the system prompt in
the Hermes convention.

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3-32b-q4_k_m",
  "stream": false,
  "messages": [{"role": "user", "content": "Save Hello at topics/hello.md"}],
  "tools": [{"type": "function", "function": {
    "name": "write_wiki", "description": "Create or overwrite a wiki article",
    "parameters": {"type": "object",
      "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
      "required": ["path", "content"]}}}]
}'
```

```json
{"message": {"role": "assistant", "content": "",
  "tool_calls": [{"function": {"name": "write_wiki",
    "arguments": {"path": "topics/hello.md", "content": "Hello"}}}]},
 "done": true}
```

When streaming, `tool_calls` arrive on the final `done: true` chunk. The
executable spec lives in `tests/test_tool_calling_acceptance.py`.

</details>

<details>
<summary><b>Text to speech</b> — Bark, SpeechT5, Coqui XTTS</summary>

```bash
pip install "hfl[tts,audio]"
hfl pull suno/bark-small --alias bark
hfl tts bark "Hello, this is a test." -o hello.wav     # write a file (wav, mp3, ogg)
hfl speak bark "Hola mundo" --lang es --speed 0.9      # play it
```

Options: `--lang`, `--voice`, `--speed` (0.25–4.0), and for `tts` also
`--output`, `--rate` and `--format`. The same voices are served over HTTP:

```bash
# OpenAI-compatible
curl http://localhost:11434/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"model": "bark", "input": "Hello world", "voice": "alloy"}' --output speech.wav

# Native: language, speed, sample rate and format (wav, mp3, ogg)
curl http://localhost:11434/api/tts -H "Content-Type: application/json" \
  -d '{"model": "bark", "text": "Hola mundo", "language": "es"}' --output speech.wav
```

</details>

<details>
<summary><b>More tools</b> — LoRA, KV snapshots, speculative decoding, MCP, Hub upload</summary>

- `hfl lora apply|remove|list` — hot-swap LoRA adapters without reloading the base model.
- `hfl snapshot save|load|list|delete` — persist the KV cache to disk for warm starts.
- `hfl draft-recommend` — pick a small Hub sibling for speculative decoding.
- `hfl mcp serve` / `hfl mcp connect` — run as a Model Context Protocol server, or use MCP tools.
- `hfl compliance-dashboard` — license risk across your local models.
- `POST /api/push` — upload a registered model to the Hub.
- `WS /ws/chat` — bidirectional chat with frame-level cancellation.

</details>

## Connect your tools

**Coding agents.** One command opens Claude Code or Codex on a local model:

```bash
hfl launch claude -m hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M
hfl launch codex  -m qwen-coder                 # a local name or alias works too
hfl launch claude -m qwen-coder --print         # just show the settings for your shell
```

HFL downloads the model if needed, starts a server if none is running (and
stops it when the agent exits), loads the model and hands the agent its real
context window. Nothing in the agent's own configuration is changed.
Arguments after `--` go to the agent: `hfl launch claude -m qwen-coder -- -p "fix the tests"`.

The server listens on `http://localhost:11434` and speaks three APIs. It answers
one request at a time per GGUF model by default; `hfl serve --parallel 4` (with
llama.cpp installed) serves several at once — what coding agents and several
users need.

**OpenAI** — `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/responses`,
`/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/images/generations`

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")
reply = client.chat.completions.create(
    model="hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",
    messages=[{"role": "user", "content": "Explain quantum computing in one paragraph"}],
)
print(reply.choices[0].message.content)
```

**Ollama** — `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/ps`, `/api/pull`,
`/api/delete` (only from the server's own machine) and more

```bash
curl http://localhost:11434/api/chat -d '{"model": "llama70b",
  "messages": [{"role": "user", "content": "Hello!"}]}'
```

**Anthropic** — `/v1/messages`

```bash
curl http://localhost:11434/v1/messages -H "Content-Type: application/json" \
  -d '{"model": "llama70b", "max_tokens": 256,
       "messages": [{"role": "user", "content": "Hello!"}]}'
```

Models can be named by their local name, an alias, or the Hub reference they
were pulled from (`hf.co/org/repo:QUANT`). The server never downloads on its
own: `hfl pull` or `hfl serve --model <reference>` does.

## Reference

<details>
<summary><b>Configuration</b></summary>

| Variable | Default | What it does |
|---|---|---|
| `HFL_HOME` | `~/.hfl` | Where models, the registry and logs live |
| `HF_TOKEN` | — | Hugging Face token for gated models (or `hfl login`) |
| `HFL_MEMORY_BUDGET` | `85` | % of total RAM that may be in use after a load |
| `HFL_KEEP_ALIVE` | `5m` | How long an idle model stays loaded (`-1` = forever) |
| `HFL_MAX_LOADED_MODELS` | `0` | Optional ceiling on the number of loaded models |
| `HFL_LANG` | `en` | CLI language: `en` or `es` |

Settings Ollama also has (`OLLAMA_HOST`, `OLLAMA_KEEP_ALIVE`,
`OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`, …) are read under either
name. The full list is in [docs/env-vars.md](https://github.com/ggalancs/hfl/blob/main/docs/env-vars.md).

Protect the API with a key: `hfl serve --api-key <secret>`, then send
`Authorization: Bearer <secret>` or `X-API-Key: <secret>`.

</details>

<details>
<summary><b>Concurrency and backpressure</b></summary>

llama.cpp and Transformers drive a single model instance that cannot take two
requests at once, so HFL runs inference one request at a time behind a bounded
queue, shared by the three APIs:

| Setting | Env var | Default |
|---|---|---|
| Requests running at once | `HFL_QUEUE_MAX_INFLIGHT` | `1` |
| Requests allowed to wait | `HFL_QUEUE_MAX_SIZE` | `16` |
| Seconds a request may wait | `HFL_QUEUE_ACQUIRE_TIMEOUT` | `60` |

A full queue answers **429** with `Retry-After`; a request that waited too long
answers **503**. Every response carries `X-Queue-Depth` and related headers, and
`GET /healthz` reports the live state.

</details>

<details>
<summary><b>Quantization levels</b></summary>

`Q4_K_M` is the default and the usual balance between size and quality. `Q5_K_M`,
`Q6_K` and `Q8_0` stay closer to the original model and take more memory; `Q3_K_M`
and `Q2_K` take less, at a visible cost in quality; `F16` is not quantized.
HFL tells you before loading whether a model fits — and `hfl recommend` suggests
the ones that do.

</details>

<details>
<summary><b>How it works</b></summary>

```text
hfl pull / run ──▶ Hugging Face Hub ──▶ ~/.hfl/models ──▶ GGUF? ── yes ──▶ llama.cpp
                   (search, download,                  MLX build (Apple Silicon) ──▶ MLX
                    license check)                     safetensors ── convert + quantize ──▶ GGUF

hfl serve ──▶ OpenAI · Ollama · Anthropic APIs ──▶ memory-budgeted model set ──▶ one inference at a time
```

The [architecture guide](https://htmlpreview.github.io/?https://github.com/ggalancs/hfl/blob/main/docs/hfl-architecture-complete.html)
covers the modules, engine selection, the conversion pipeline and every endpoint
([en español](https://htmlpreview.github.io/?https://github.com/ggalancs/hfl/blob/main/docs/hfl-arquitectura-completa.html)).

</details>

## Documentation

- [Hub-native features](https://github.com/ggalancs/hfl/blob/main/docs/hub-native-features.md) — discover, recommend, pull-smart, verify, bench and more
- [Environment variables](https://github.com/ggalancs/hfl/blob/main/docs/env-vars.md) — every setting and its default
- [Apple Silicon and Docker clients](https://github.com/ggalancs/hfl/blob/main/docs/apple-silicon-and-docker-clients.md)
- [Benchmarks](https://github.com/ggalancs/hfl/blob/main/docs/benchmarks.md) — HFL, Ollama and llama-server on the same GGUF, with the script to rerun it
- [Architecture guide](https://htmlpreview.github.io/?https://github.com/ggalancs/hfl/blob/main/docs/hfl-architecture-complete.html)
- [Changelog](https://github.com/ggalancs/hfl/blob/main/CHANGELOG.md)

**Status:** beta — 4,000+ tests at ~90% coverage. Windows builds and installers
are published, but Windows is less tested than macOS and Linux.

## Contributing

Issues and pull requests are welcome — see [CONTRIBUTING.md](https://github.com/ggalancs/hfl/blob/main/CONTRIBUTING.md).

```bash
git clone https://github.com/ggalancs/hfl && cd hfl
pip install -e ".[dev]"
bash scripts/ci-local.sh        # lint, types and the full test suite, as CI runs them
```

If HFL saves you a download–convert–quantize afternoon, a ⭐ helps other people find it.

## Legal notices

**Model licenses.** Models keep their own licenses (Llama, Gemma, OpenRAIL,
CC-BY-NC, …) and you are responsible for complying with them. HFL shows a
model's license before downloading it, stores it with the model and records the
pull's provenance — see `hfl inspect <model>`. Common restrictions include
non-commercial use only (CC-BY-NC, MRL), attribution (Llama, Gemma) and usage
restrictions (OpenRAIL).

**Export compliance.** HFL only downloads publicly available open-weight models
from the Hugging Face Hub and does not facilitate access to closed-weight or
export-controlled weights. Users are responsible for complying with the export
regulations of their jurisdiction.

**Disclaimer.** AI models may generate inaccurate, biased or inappropriate
content. Users are solely responsible for evaluating and using model outputs.
See [DISCLAIMER.md](https://github.com/ggalancs/hfl/blob/main/DISCLAIMER.md).

**Trademarks.** "OpenAI" is a trademark of OpenAI, Inc. "Ollama" is a trademark
of Ollama, Inc. "Anthropic" is a trademark of Anthropic, PBC. "Hugging Face" and
the Hugging Face logo are trademarks of Hugging Face, Inc. These marks are used
for identification only. **HFL is an independent project, not affiliated with,
endorsed by or officially connected to any of these companies.** References to
their services describe technical interoperability only.

## License

HFL is licensed under the **Apache License 2.0** — you may use, modify,
distribute and sell it, including commercially, as long as you keep the
copyright and license notices. See [LICENSE](https://github.com/ggalancs/hfl/blob/main/LICENSE) and [NOTICE](https://github.com/ggalancs/hfl/blob/main/NOTICE).

HFL ships responsible-use safeguards: license checking, AI disclaimers,
provenance tracking, privacy protections and respect for gated models.
Apache-2.0 does not require you to keep them; as a project norm we ask that
redistributions leave them active. See [DISCLAIMER.md](https://github.com/ggalancs/hfl/blob/main/DISCLAIMER.md),
[PRIVACY.md](https://github.com/ggalancs/hfl/blob/main/PRIVACY.md) and [NOTICE-EU-AI-ACT.md](https://github.com/ggalancs/hfl/blob/main/NOTICE-EU-AI-ACT.md).

HFL's license covers HFL itself, not the models you download.
