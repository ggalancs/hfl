# Using HFL with other tools

HFL speaks the Ollama, OpenAI and Anthropic APIs on one port, so a tool that
supports any of them can use it. Each recipe below was **run for real** on
2026-09-28 (HFL 0.22.1 + fixes, macOS on Apple Silicon, Docker Desktop):
what is written was checked, and what was not is said.

Start HFL first:

```bash
hfl serve                 # http://127.0.0.1:11434
hfl pull Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF -q Q4_K_M --alias coder
hfl pull nomic-ai/nomic-embed-text-v1.5-GGUF -q Q4_K_M --alias embed
```

Any local model name or alias works in place of `coder` / `embed`
(`hfl list` shows them).

**Model size matters for tools.** The checks below used a 1.5B model. It
answers, writes code and calls a single tool, but faced with many tools at
once (Open WebUI offers 35) it often calls one more tool than needed or
describes the result instead of answering. For agents and tool use, prefer
7B or larger.

## Open WebUI

Checked with Open WebUI 0.11.4 (Docker): model list, chat, the titles,
tags and follow-ups it generates, and its built-in tools (the model's
calls reach Open WebUI and it runs them).

```bash
docker run -d --name open-webui -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v open-webui:/app/backend/data ghcr.io/open-webui/open-webui:main
```

- `--add-host=host.docker.internal:host-gateway` is needed: without it the
  name did not resolve inside the container.
- With a small model, turn off Open WebUI's built-in tools for it (the
  "Builtin Tools" option in the model's settings), or use a larger model.
- Qwen2.5-Coder's own GGUF template shows the tool-call format with doubled
  braces, and the model copies them; HFL reads those calls (before this fix,
  4 of 10 replies to Open WebUI arrived as text instead of a call).

## AnythingLLM

Checked with the `mintplexlabs/anythingllm` image (2026-09-28): a document
uploaded, embedded through HFL, and a question in *query* mode answered
from it, with the document cited.

```bash
docker run -d --name anythingllm -p 3001:3001 --cap-add SYS_ADMIN \
  --add-host=host.docker.internal:host-gateway \
  -e STORAGE_DIR=/app/server/storage \
  -e LLM_PROVIDER=ollama -e OLLAMA_BASE_PATH=http://host.docker.internal:11434 \
  -e OLLAMA_MODEL_PREF=coder -e OLLAMA_MODEL_TOKEN_LIMIT=4096 \
  -e EMBEDDING_ENGINE=ollama -e EMBEDDING_BASE_PATH=http://host.docker.internal:11434 \
  -e EMBEDDING_MODEL_PREF=embed -e EMBEDDING_MODEL_MAX_CHUNK_LENGTH=2048 \
  -e VECTOR_DB=lancedb \
  -v anythingllm:/app/server/storage mintplexlabs/anythingllm
```

## Continue (VS Code, JetBrains)

Checked with the Continue CLI 1.5.47 (`cn -p`), which reads the same
`config.yaml` as the editor extensions: both providers below answered.

```yaml
# ~/.continue/config.yaml
name: HFL
version: 1.0.0
schema: v1
models:
  - name: HFL coder
    provider: ollama
    model: coder
    apiBase: http://127.0.0.1:11434
    roles: [chat, edit, apply]
  - name: HFL coder (OpenAI API)
    provider: openai
    model: coder
    apiBase: http://127.0.0.1:11434/v1
    apiKey: unused
    roles: [chat]
```

The editor extensions themselves were not driven in this check.

## aider

Checked with aider 0.86.2 in a git repository: asked to rename a function,
it edited the file as asked.

```bash
export OLLAMA_API_BASE=http://127.0.0.1:11434
aider --model ollama_chat/coder --edit-format whole
```

`--edit-format whole` suits small models; with a 0.5B model aider talked to
HFL fine but the model wrote a new file instead of editing the one asked.

## LangChain

Checked with langchain-openai 1.6.6, langchain-ollama 1.1.0: chat, streaming,
tool calling (`bind_tools`) and embeddings, through either API.

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
llm = ChatOpenAI(base_url="http://127.0.0.1:11434/v1", api_key="unused", model="coder")
emb = OpenAIEmbeddings(base_url="http://127.0.0.1:11434/v1", api_key="unused", model="embed")

from langchain_ollama import ChatOllama, OllamaEmbeddings
llm = ChatOllama(base_url="http://127.0.0.1:11434", model="coder")
emb = OllamaEmbeddings(base_url="http://127.0.0.1:11434", model="embed")
```

`OpenAIEmbeddings` sends token ids instead of text by default; HFL accepts
both.

## LiteLLM

Checked with LiteLLM 1.103.0, through both of its providers:

```python
import litellm
litellm.completion(model="ollama_chat/coder", api_base="http://127.0.0.1:11434", messages=[...])
litellm.completion(model="openai/coder", api_base="http://127.0.0.1:11434/v1",
                   api_key="unused", messages=[...])
```

## Coding agents: Claude Code, Codex

`hfl launch claude -m <model>` and `hfl launch codex -m <model>` start them
on a local model; see the README.

## Not checked

- **LibreChat**: it needs MongoDB and more services; not run.
- **Docker on Linux**: the recipes above ran on Docker Desktop for macOS,
  which reaches an `hfl serve` bound to `127.0.0.1`. On Linux a container
  does not reach the host's loopback: run `hfl serve --host 0.0.0.0` with
  `HFL_API_KEY` set (and give the tool the key), or run the tool with
  `--network=host`. Not run here.
- Running HFL itself in a container on macOS loses the GPU: see
  [apple-silicon-and-docker-clients.md](apple-silicon-and-docker-clients.md).
