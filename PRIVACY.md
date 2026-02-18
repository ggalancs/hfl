# Privacy Policy

## Data Collection

hfl does **NOT** collect, store, or transmit any user data to external servers.

All processing occurs locally on your machine.

## HuggingFace Tokens

hfl offers two methods for HuggingFace authentication, with different privacy tradeoffs:

### Option 1: `hfl login` (Convenient)

```bash
hfl login
```

- Uses `huggingface_hub` library's official login mechanism
- Token is stored by huggingface_hub in `~/.cache/huggingface/token`
- Persists across sessions (no need to re-enter)
- Same storage used by `huggingface-cli login`
- Remove with `hfl logout`

### Option 2: `HF_TOKEN` Environment Variable (More Private)

```bash
export HF_TOKEN=hf_your_token_here
hfl pull meta-llama/...
```

- Token exists only in memory during the shell session
- **NEVER** written to disk by hfl
- More secure for sensitive environments
- Must be set each session (or in shell profile)

### Token Transmission

Regardless of method:
- Tokens are **ONLY** transmitted to `api.huggingface.co` and `huggingface.co`
- Tokens are **NEVER** logged or stored in hfl's own files (`~/.hfl/`)
- Tokens are **NEVER** included in model metadata or provenance records

## User Prompts

- Chat messages are held in memory during the session only
- Messages are **NEVER** written to disk
- Messages are **NEVER** transmitted to any external server
- When the session ends, all messages are discarded
- No conversation history is retained between sessions

## API Server Logs

When running `hfl serve`:

- The API server logs request metadata (timestamp, endpoint, status code, duration)
- Request bodies (prompts) are **NEVER** logged
- Response bodies (model outputs) are **NEVER** logged
- Logs are stored locally and are not transmitted externally
- IP addresses in logs are from localhost connections only (default binding: `127.0.0.1`)

## Model Downloads

When downloading models from HuggingFace:

- hfl sends requests to `huggingface.co` and `api.huggingface.co`
- These requests include:
  - Model repository ID
  - Your HuggingFace token (if provided)
  - hfl version in User-Agent header
- HuggingFace's privacy policy applies to their servers

## Telemetry

- hfl does **NOT** include any telemetry, analytics, or tracking mechanisms
- hfl does **NOT** "phone home" or make any network requests except to HuggingFace for model downloads
- No crash reports or usage statistics are collected

## Local Storage

hfl stores the following locally:

| Location | Contents |
|----------|----------|
| `~/.hfl/models/` | Downloaded model files |
| `~/.hfl/models.json` | Registry of downloaded models (names, paths, metadata) |
| `~/.hfl/provenance.json` | Record of model conversions (for legal compliance) |
| `~/.hfl/config.yaml` | User configuration (if created) |

If you use `hfl login`, the HuggingFace token is stored by the huggingface_hub library:

| Location | Contents |
|----------|----------|
| `~/.cache/huggingface/token` | HuggingFace token (if `hfl login` was used) |

This file can be removed with `hfl logout`.

**hfl's own files (`~/.hfl/`) never contain:**
- User prompts or conversations
- Authentication tokens
- Personal information
- Usage statistics

## Third-Party Services

hfl interacts with:

| Service | Purpose | Data Sent |
|---------|---------|-----------|
| HuggingFace Hub | Model downloads | Repository ID, auth token, User-Agent |

hfl does **NOT** interact with OpenAI, Anthropic, Google, or any other AI service providers.

## Your Rights

Since hfl does not collect personal data, there is no data to:
- Access
- Correct
- Delete
- Export

If you have concerns about HuggingFace's data practices, please refer to their privacy policy at https://huggingface.co/privacy

## Contact

For privacy-related questions about hfl, please open an issue on the project repository.

---

*Last updated: February 2026*
