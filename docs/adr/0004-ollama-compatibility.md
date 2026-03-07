# ADR-0004: Ollama API Compatibility

## Status

Accepted

## Context

Many tools and applications are built to work with Ollama's API. To enable HFL as a drop-in replacement for Ollama while providing access to HuggingFace's model ecosystem, we need API compatibility.

Goals:
- Existing Ollama clients work unchanged
- OpenAI SDK compatibility for broader ecosystem
- Native HFL features accessible
- Minimal API surface to maintain

## Decision

We implement both Ollama-native API and OpenAI-compatible API endpoints:

**Ollama-native endpoints (primary compatibility):**
- `POST /api/generate` - Text generation
- `POST /api/chat` - Chat completion
- `GET /api/tags` - List models
- `POST /api/pull` - Download model
- `DELETE /api/delete` - Remove model
- `GET /api/version` - Server version

**OpenAI-compatible endpoints:**
- `POST /v1/chat/completions` - Chat (streaming/non-streaming)
- `POST /v1/completions` - Text completion
- `GET /v1/models` - List models

**Request/response mapping:**
```python
# Ollama format
{
    "model": "llama2",
    "messages": [...],
    "options": {"temperature": 0.7}
}

# OpenAI format
{
    "model": "llama2",
    "messages": [...],
    "temperature": 0.7
}
```

## Consequences

### Positive

- Drop-in replacement for Ollama
- Works with OpenAI SDK
- Broad ecosystem compatibility
- Users don't need to learn new API

### Negative

- Two API formats to maintain
- Some Ollama features not implemented (embeddings, multimodal)
- Response format differences to handle
- Test coverage for both APIs

### Neutral

- Default port 11434 (same as Ollama)
- Model naming follows HuggingFace convention

## Alternatives Considered

### Option A: Ollama-Only API

Only implement Ollama API format.

**Pros:**
- Simpler implementation
- Single API to maintain

**Cons:**
- Misses OpenAI ecosystem
- Some tools only support OpenAI

### Option B: OpenAI-Only API

Only implement OpenAI API format.

**Pros:**
- Broader SDK support
- Industry standard

**Cons:**
- Not a drop-in Ollama replacement
- Different default behaviors

### Option C: Custom HFL API

Design a new API optimized for HFL.

**Pros:**
- Optimal for our use cases
- No compatibility constraints

**Cons:**
- No ecosystem support
- Users need to learn new API
- More documentation needed

## References

- Ollama API documentation
- OpenAI API reference
- Local LLM API comparison
