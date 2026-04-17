# hfl

[![License: HRUL v1.0](https://img.shields.io/badge/License-HRUL%20v1.0-blue.svg)](LICENSE) [![License FAQ](https://img.shields.io/badge/License-FAQ-informational.svg)](LICENSE-FAQ.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/ggalancs/hfl/actions/workflows/ci.yml/badge.svg)](https://github.com/ggalancs/hfl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ggalancs/hfl/branch/main/graph/badge.svg)](https://codecov.io/gh/ggalancs/hfl)

Run HuggingFace models locally like Ollama.

> **[Versión en Español](README.es.md)**

## Why HFL?

**Ollama has a curated catalog of ~500 models. HuggingFace Hub has 500,000+.**

If you want to run a model that isn't in Ollama's catalog — a specific fine-tune, a recent release from a small lab, a niche model — you have to manually download from HuggingFace, convert to GGUF with llama.cpp, quantize, and configure inference. **HFL automates all of this in a single command.**

| Feature | Ollama | HFL |
|---------|--------|-----|
| Model catalog | ~500 curated | 500K+ (all HF Hub) |
| Auto-conversion | Not needed (pre-converted) | Yes (safetensors→GGUF) |
| Ease of use | Excellent | Good |
| OpenAI API compatible | Yes | Yes |
| Ollama API compatible | Native | Yes (drop-in) |
| Anthropic Messages API | No | Yes (Claude Code compatible) |
| Structured tool calling | Yes | Yes (qwen / llama3 / mistral) |
| Multi-backend | llama.cpp only | llama.cpp + Transformers + vLLM |
| License verification | No | Yes (5 risk levels) |
| Legal traceability | No | Yes (provenance log) |
| Maturity | High (established) | Alpha (v0.3.5) |

**HFL doesn't compete with Ollama — it complements it.** Use Ollama for curated models; use HFL when you need something from the full HuggingFace ecosystem.

## Features

- **CLI & API**: Full CLI interface plus REST API compatible with OpenAI, Ollama, and Anthropic
- **Model Search**: Interactive paginated search of HuggingFace Hub (like `more`)
- **Multiple Backends**: llama.cpp (GGUF/CPU), Transformers (GPU native), vLLM (production)
- **Automatic Conversion**: Downloads HuggingFace models and converts to GGUF automatically
- **Smart Quantization**: Supports Q2_K through F16 quantization levels
- **Text-to-Speech**: Native TTS support with Bark, SpeechT5, Coqui XTTS and more
- **Structured tool calling**: Ollama-compatible `tools` / `tool_calls` wire protocol with per-family parsers for qwen, llama3, and mistral — agents work out of the box
- **Bounded inference queue**: server-side serialisation of requests with explicit 429 / 503 backpressure, live `X-Queue-Depth` headers, and `GET /healthz` for orchestrators
- **Drop-in Compatible**: Works as a replacement for Ollama with existing tooling
- **Internationalized**: Full i18n support (English, Spanish) - set `HFL_LANG` to change language

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HFL Architecture                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐         ┌──────────────────┐         ┌─────────────────┐   │
│  │  hfl pull   │───────▶ │  HuggingFace Hub │───────▶ │  Local Storage  │   │
│  │             │         │                  │         │   ~/.hfl/       │   │
│  └─────────────┘         │  • Search API    │         │   ├── models/   │   │
│        │                 │  • Download      │         │   ├── cache/    │   │
│        │                 │  • License info  │         │   └── registry  │   │
│        ▼                 └──────────────────┘         └─────────────────┘   │
│  ┌─────────────┐                                              │             │
│  │  Converter  │◀─────────────────────────────────────────────┘             │
│  │             │                                                            │
│  │ safetensors │──────────▶ GGUF (quantized Q2_K...F16)                     │
│  └─────────────┘                                                            │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐         ┌──────────────────┐         ┌─────────────────┐   │
│  │  hfl run    │───────▶ │ Inference Engine │───────▶ │  Interactive    │   │
│  │             │         │                  │         │     Chat        │   │
│  └─────────────┘         │  • llama.cpp     │         └─────────────────┘   │
│                          │  • Transformers  │                               │
│                          │  • vLLM          │                               │
│                          └──────────────────┘                               │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐         ┌──────────────────┐         ┌─────────────────┐   │
│  │  hfl serve  │───────▶ │   REST API       │───────▶ │  OpenAI SDK /   │   │
│  │             │         │                  │         │  Ollama clients │   │
│  └─────────────┘         │  • /v1/chat/...  │         └─────────────────┘   │
│                          │  • /api/chat     │                               │
│                          │  • /api/generate │                               │
│                          └──────────────────┘                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Flow Summary:**
1. **Pull**: Download from HuggingFace Hub → Convert to GGUF (if needed) → Store locally
2. **Run**: Load model into inference engine → Start interactive chat session
3. **Serve**: Start API server → Accept OpenAI/Ollama-compatible requests

## Prerequisites

- **Python 3.10+** (required)
- **git** (for cloning llama.cpp during first conversion)
- **cmake** and **C++ compiler** (for building llama.cpp quantization tools)
  - macOS: `xcode-select --install`
  - Ubuntu/Debian: `sudo apt install build-essential cmake`
  - Windows: Install Visual Studio Build Tools

> **Note:** Build tools are only needed if you convert safetensors models to GGUF. If you only use pre-quantized GGUF models, they're not required.

## Installation

```bash
# Clone the repository
git clone https://github.com/ggalancs/hfl
cd hfl

# Basic installation (CPU + GGUF)
pip install .

# With GPU support (Transformers + bitsandbytes)
pip install ".[transformers]"

# With TTS support (Bark, SpeechT5)
pip install ".[tts]"

# With Coqui TTS (XTTS-v2, VITS)
pip install ".[coqui]"

# With vLLM for production
pip install ".[vllm]"

# Everything
pip install ".[all]"
```

## Quick Start

### Download a Model

```bash
# Download with default Q4_K_M quantization
hfl pull meta-llama/Llama-3.3-70B-Instruct

# Specify quantization level
hfl pull meta-llama/Llama-3.3-70B-Instruct --quantize Q5_K_M

# Keep as safetensors (for GPU inference)
hfl pull mistralai/Mistral-7B-Instruct-v0.3 --format safetensors

# Download with a custom alias for easier reference
hfl pull meta-llama/Llama-3.3-70B-Instruct --alias llama70b
```

### Interactive Chat

```bash
# Start chat with a model
hfl run llama-3.3-70b-instruct-q4_k_m

# With system prompt
hfl run llama-3.3-70b-instruct-q4_k_m --system "You are a Python expert"
```

### API Server

```bash
# Start server (default port 11434, same as Ollama)
hfl serve

# Pre-load a model
hfl serve --model llama-3.3-70b-instruct-q4_k_m

# Custom host/port
hfl serve --host 0.0.0.0 --port 8080
```

### Text-to-Speech (TTS)

HFL supports TTS models from HuggingFace like Bark, SpeechT5, and Coqui XTTS.

```bash
# Download a TTS model (no GGUF conversion needed)
hfl pull suno/bark-small --alias bark

# Synthesize text to audio file
hfl tts bark "Hello, this is a test." -o output.wav

# Synthesize and play directly (requires sounddevice)
hfl speak bark "Hello, this is a test."

# With options
hfl tts bark "Hola mundo" --lang es --output spanish.wav --speed 0.9
hfl speak bark "Fast speech" --speed 1.5
```

**TTS Options:**
- `--output, -o`: Output file path (default: output.wav)
- `--lang, -l`: Language code (en, es, fr, etc.)
- `--voice, -v`: Voice/speaker to use
- `--speed, -s`: Speed multiplier (0.25-4.0)
- `--rate, -r`: Sample rate in Hz
- `--format, -f`: Audio format (wav, mp3, ogg)

**TTS API:**
```bash
# OpenAI-compatible endpoint
curl -X POST http://localhost:11434/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "bark", "input": "Hello world", "voice": "alloy"}' \
  --output speech.wav

# Native HFL endpoint
curl -X POST http://localhost:11434/api/tts \
  -H "Content-Type: application/json" \
  -d '{"model": "bark", "text": "Hello world", "language": "en"}' \
  --output speech.wav
```

### Search Models on HuggingFace

```bash
# Search for models (paginated like 'more')
hfl search llama

# Search only models with GGUF files
hfl search mistral --gguf

# Customize pagination and results
hfl search phi --limit 50 --page-size 5

# Sort by likes instead of downloads
hfl search qwen --sort likes
```

**Navigation controls:**
- `SPACE` / `ENTER` - Next page
- `p` - Previous page
- `q` / `ESC` - Exit

### Model Management

```bash
# List all local models
hfl list

# Show model details
hfl inspect llama-3.3-70b-instruct-q4_k_m

# Remove a model
hfl rm llama-3.3-70b-instruct-q4_k_m

# Set an alias for an existing model
hfl alias llama-3.3-70b-instruct-q4_k_m llama70b

# Now use the alias in any command
hfl run llama70b
hfl inspect llama70b
```

## API Endpoints

### OpenAI-Compatible

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b-instruct-q4_k_m",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Ollama-Compatible

```bash
curl http://localhost:11434/api/chat \
  -d '{
    "model": "llama-3.3-70b-instruct-q4_k_m",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Using OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="llama-3.3-70b-instruct-q4_k_m",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
)
print(response.choices[0].message.content)
```

## Tool Calling (Agents)

HFL implements the Ollama wire protocol for **structured tool calling**, so
agents written against the Ollama Python SDK can drive multi-turn tool loops
directly. When a client sends `tools` on `/api/chat`, HFL forwards them
through the model's native chat template (qwen3 `<tool_call>`, llama3
`<|python_tag|>`, mistral `[TOOL_CALLS]`), parses the reply into canonical
`message.tool_calls` with `arguments` as a parsed object, and accepts
`role: "tool"` results on the next turn.

```bash
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-32b-q4_k_m",
    "stream": false,
    "messages": [
      {"role":"system","content":"You MUST call write_wiki, never respond with text."},
      {"role":"user","content":"Save Hello at topics/hello.md"}
    ],
    "tools":[{
      "type":"function",
      "function":{
        "name":"write_wiki",
        "description":"Create or overwrite a wiki article",
        "parameters":{
          "type":"object",
          "properties":{"path":{"type":"string"},"content":{"type":"string"}},
          "required":["path","content"]
        }
      }
    }]
  }'
```

Response:

```json
{
  "model": "qwen3-32b-q4_k_m",
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "function": {
          "name": "write_wiki",
          "arguments": {"path": "topics/hello.md", "content": "Hello"}
        }
      }
    ]
  },
  "done": true
}
```

The per-family parser also handles a generic `{"tool_call": {...}}`
fallback for templates that weren't properly applied. Streaming (`stream:
true`) accumulates the full reply and emits `tool_calls` on the final
`done: true` chunk. Full spec is in
[`hfl-tool-calling-spec.md`](../hfl-tool-calling-spec.md) and the
acceptance suite lives at `tests/test_tool_calling_acceptance.py`.

## Concurrency & Backpressure

Local inference backends (llama.cpp, transformers-GPU) share a single
non-reentrant model instance. HFL protects them with an **in-server
inference dispatcher** that serialises requests on the currently-loaded
engine with a bounded wait queue:

| Setting | Env var | Default | Meaning |
|---|---|---|---|
| Max in-flight | `HFL_QUEUE_MAX_INFLIGHT` | `1` | Parallel requests allowed |
| Wait queue size | `HFL_QUEUE_MAX_SIZE` | `16` | Requests allowed to wait |
| Acquire timeout | `HFL_QUEUE_ACQUIRE_TIMEOUT` | `60` | Seconds a request may wait |
| Enabled | `HFL_QUEUE_ENABLED` | `true` | Master switch |

When the wait queue is saturated, HFL returns **429** with a structured
envelope and `Retry-After`:

```json
{
  "error": "Inference queue is full",
  "code": "QUEUE_FULL",
  "category": "rate_limit",
  "retryable": true,
  "details": {"retry_after_seconds": 60, "queue_depth": 1, "max_queued": 1}
}
```

When a caller has been queued longer than `HFL_QUEUE_ACQUIRE_TIMEOUT`,
HFL returns **503** with `code=QUEUE_TIMEOUT`. Every response carries
`X-Queue-Depth`, `X-Queue-In-Flight`, `X-Queue-Max-Inflight` and
`X-Queue-Max-Size` so agents can back off proportionally. Live state is
also available via:

```bash
curl http://localhost:11434/healthz
# { "status":"ok", "models_loaded":[...], "queue_depth":0,
#   "queue_in_flight":0, "uptime_seconds":12345 }
```

All three API surfaces (Ollama, OpenAI, Anthropic) share the same
dispatcher, so a slow call on `/api/chat` correctly blocks
`/v1/chat/completions` and `/v1/messages`.

## Quantization Levels

| Level | Bits/weight | Quality | Use Case |
|-------|-------------|---------|----------|
| Q2_K | ~2.5 | ~80% | Extreme compression |
| Q3_K_M | ~3.5 | ~87% | Low RAM |
| **Q4_K_M** | ~4.5 | ~92% | **Default - best balance** |
| Q5_K_M | ~5.0 | ~96% | High quality |
| Q6_K | ~6.5 | ~97% | Premium |
| Q8_0 | ~8.0 | ~98%+ | Maximum quantized quality |
| F16 | 16.0 | 100% | No quantization |

## RAM Requirements

```
RAM needed ≈ (parameters × bits_per_weight) / 8 + 2GB overhead

Example: Llama 3.3 70B with Q4_K_M
= (70B × 4.5) / 8 + 2GB ≈ 41.4 GB
```

| Model Size | Q4_K_M RAM | Recommended Hardware |
|------------|------------|---------------------|
| 7B | ~5 GB | 8 GB RAM |
| 13B | ~9 GB | 16 GB RAM |
| 30B | ~20 GB | 32 GB RAM |
| 70B | ~42 GB | 48 GB+ RAM or GPU |

## Authentication

Configure your HuggingFace token for faster downloads and access to gated models:

```bash
# Interactive login (recommended - stores token securely)
hfl login

# Or use environment variable (more private - not persisted)
export HF_TOKEN=hf_your_token_here
```

Get your token at: https://huggingface.co/settings/tokens

## Configuration

Environment variables:
- `HFL_HOME`: Data directory (default: `~/.hfl`)
- `HF_TOKEN`: HuggingFace token for gated models (alternative to `hfl login`)
- `HFL_LANG`: Interface language (`en` for English, `es` for Spanish). Defaults to English.

### Language Support

hfl supports multiple languages. Set the `HFL_LANG` environment variable to change the CLI language:

```bash
# Use Spanish
export HFL_LANG=es
hfl --help

# Use English (default)
export HFL_LANG=en
hfl --help
```

Supported languages: English (`en`), Spanish (`es`)

## Known Limitations

This is a v0.3.x alpha release. Known limitations include:

- **vLLM backend is experimental**: Basic implementation without full streaming support
- **CORS is restrictive by default**: same-origin only; opt in via `cors_allow_all` or explicit `cors_origins`
- **Windows support**: Not fully tested; Unix-like systems recommended

### API Authentication

The API server supports optional authentication via the `--api-key` flag:

```bash
# Start server with authentication
hfl serve --api-key your-secret-key

# Client requests must include the key
curl -H "Authorization: Bearer your-secret-key" http://localhost:11434/v1/models
# Or
curl -H "X-API-Key: your-secret-key" http://localhost:11434/v1/models
```

## Documentation

Complete architecture documentation with diagrams is available:

- **[📖 View Architecture Documentation](https://htmlpreview.github.io/?https://github.com/ggalancs/hfl/blob/main/docs/hfl-architecture-complete.html)** - Interactive HTML documentation with architecture diagrams, module descriptions, and flow charts

The documentation covers:
- System architecture and design patterns
- Module structure and dependencies
- Inference engine selection logic
- GGUF conversion pipeline
- Legal compliance features
- API endpoints reference

> **Note:** Documentation is also available in [Spanish](https://htmlpreview.github.io/?https://github.com/ggalancs/hfl/blob/main/docs/hfl-arquitectura-completa.html).

## Development

```bash
# Clone and install in development mode
git clone https://github.com/ggalancs/hfl
cd hfl
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=hfl --cov-report=term-missing

# Format code
ruff format .
ruff check . --fix
```

## Legal Notices

### Export Compliance

hfl only downloads publicly available open-weight models from HuggingFace Hub. Users are responsible for compliance with applicable export control regulations in their jurisdiction.

hfl does not facilitate access to closed-weight or export-controlled model weights.

### Model Licenses

Models downloaded through hfl may have their own license restrictions. hfl displays license information before download and stores it with the model metadata. Users are responsible for complying with model licenses.

Common restrictions include:
- **Non-commercial use only** (CC-BY-NC, MRL)
- **Attribution required** (Llama, Gemma)
- **Usage restrictions** (OpenRAIL)

Use `hfl inspect <model>` to view license details for downloaded models.

### Disclaimer

AI models may generate inaccurate, biased, or inappropriate content. Users are solely responsible for evaluating and using model outputs appropriately. See [DISCLAIMER.md](DISCLAIMER.md) for full details.

## Trademarks

"OpenAI" is a trademark of OpenAI, Inc. "Ollama" is a trademark of Ollama, Inc. "Hugging Face" and the Hugging Face logo are trademarks of Hugging Face, Inc. These marks are used here for identification purposes only.

**hfl is an independent project and is not affiliated with, endorsed by, or officially connected to Hugging Face, Inc., OpenAI, Inc., or Ollama, Inc.** References to these services describe technical interoperability only.

## License

hfl is source-available under the **hfl Responsible Use License (HRUL) v1.0**.

This license allows free use, modification, and commercial distribution with one condition: derivative works that are publicly distributed must maintain the legal compliance features (license checking, AI disclaimers, provenance tracking, privacy protections, and gating respect).

You are free to rewrite, extend, rebrand, and sell derivatives — you just can't strip out the safety features.

**Note:** The HRUL is not an OSI-approved open-source license. It is a source-available license with responsible use requirements, inspired by Apache 2.0, GPL copyleft, and the RAIL family of AI licenses.

See [LICENSE](LICENSE) for the full text and [LICENSE-FAQ.md](LICENSE-FAQ.md) for common questions.
