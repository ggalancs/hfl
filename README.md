# hfl

[![License: HRUL v1.0](https://img.shields.io/badge/License-HRUL%20v1.0-blue.svg)](LICENSE) [![License FAQ](https://img.shields.io/badge/License-FAQ-informational.svg)](LICENSE-FAQ.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/ggalancs/hfl/actions/workflows/ci.yml/badge.svg)](https://github.com/ggalancs/hfl/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-81%25-brightgreen.svg)](https://github.com/ggalancs/hfl)

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
| Multi-backend | llama.cpp only | llama.cpp + Transformers + vLLM |
| License verification | No | Yes (5 risk levels) |
| Legal traceability | No | Yes (provenance log) |
| Maturity | High (established) | Pre-alpha (v0.1.0) |

**HFL doesn't compete with Ollama — it complements it.** Use Ollama for curated models; use HFL when you need something from the full HuggingFace ecosystem.

## Features

- **CLI & API**: Full CLI interface plus REST API compatible with OpenAI and Ollama
- **Model Search**: Interactive paginated search of HuggingFace Hub (like `more`)
- **Multiple Backends**: llama.cpp (GGUF/CPU), Transformers (GPU native), vLLM (production)
- **Automatic Conversion**: Downloads HuggingFace models and converts to GGUF automatically
- **Smart Quantization**: Supports Q2_K through F16 quantization levels
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
# Basic installation (CPU + GGUF)
pip install hfl

# With GPU support (Transformers + bitsandbytes)
pip install "hfl[transformers]"

# With vLLM for production
pip install "hfl[vllm]"

# Everything
pip install "hfl[all]"
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

This is a v0.1.0 release. Known limitations include:

- **vLLM backend is experimental**: Basic implementation without full streaming support
- **No rate limiting on API**: Only HuggingFace Hub calls are rate-limited
- **CORS is permissive**: API allows all origins (configurable in future versions)
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
