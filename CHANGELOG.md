# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/ggalancs/hfl/releases/tag/v0.1.0
