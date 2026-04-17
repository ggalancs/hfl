# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Model capability detection (Ollama-compatible).

Computes the ``capabilities`` field emitted by ``POST /api/show`` and
by the HFL ``inspect`` / ``show`` CLI. This list is what Ollama SDK
clients (``ollama-python``, Open WebUI, LangChain) use to decide
whether a model supports tool calling, vision, embeddings or hidden
reasoning channels — passing the wrong list silently breaks those
flows.

Capability taxonomy (matches Ollama as of April 2026):

=============  =========================================================
Capability     Meaning
=============  =========================================================
completion     Can produce text completions. Every supported LLM has it.
tools          Emits native tool-call markers for its family. HFL's
               ``tool_parsers.dispatch`` recognises the family.
insert         Fill-in-the-middle / suffix support. Driven by the code
               family (CodeLlama, CodeGemma, StarCoder, Qwen-Coder,
               DeepSeek-Coder) — empirically the set that ships FIM
               tokens.
vision         Accepts images in chat messages. GGUF declares ``clip.*``
               keys or architecture is a known VL family.
embedding      Produces vector embeddings (not text). Architecture
               belongs to the embedding families (bert, nomic, jina, bge).
thinking       Has a visible chain-of-thought channel that HFL can
               optionally expose. Gemma 4, DeepSeek-R1, Qwen3-Thinking,
               GPT-OSS.
=============  =========================================================

Detection is deliberately permissive: substring matches on the
architecture / name fields, no strict version pinning. A false
positive on ``tools`` just makes the client attempt a tool call that
the model ignores — graceful degradation. A false negative would
silently disable the feature, which is worse, so we err toward
opting in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from hfl.models.manifest import ModelManifest


# ----------------------------------------------------------------------
# Family signatures
# ----------------------------------------------------------------------

# Families that HFL can parse native tool calls for today (see
# ``hfl.api.tool_parsers.dispatch``). Keep this in sync with that
# dispatcher; the source of truth for *parsing* lives there.
_TOOL_CAPABLE_FAMILIES = {
    "qwen",
    "llama3",
    "llama",
    "mistral",
    "gemma4",
    "gemma-4",
    "gemma 4",
    "mixtral",
}

# Families that ship fill-in-the-middle tokens.
_INSERT_CAPABLE_FAMILIES = {
    "codellama",
    "code-llama",
    "codegemma",
    "code-gemma",
    "starcoder",
    "starcoder2",
    "qwen-coder",
    "qwen2-coder",
    "qwen2.5-coder",
    "deepseek-coder",
    "deepseek_coder",
}

# Vision-language families that accept image inputs. A manifest is ALSO
# treated as vision-capable when its file_hash/path references CLIP
# projector files — we pick that up separately.
_VISION_FAMILIES = {
    "llava",
    "bakllava",
    "llama-3.2-vision",
    "llama3.2-vision",
    "llama4",
    "llama-4",
    "gemma-3",
    "gemma3",
    "qwen2-vl",
    "qwen2.5-vl",
    "internvl",
    "minicpm-v",
    "idefics",
    "pixtral",
    "molmo",
}

# Embedding-first architectures. These models emit vectors, not tokens.
_EMBEDDING_FAMILIES = {
    "bert",
    "nomic-bert",
    "nomic_bert",
    "nomic-embed",
    "nomic-embed-text",
    "jina-bert",
    "jina_bert",
    "jina-embeddings",
    "bge",
    "bge-small",
    "bge-large",
    "bge-m3",
    "gte",
    "e5",
    "mxbai-embed",
    "stella",
    "arctic-embed",
    "snowflake-arctic-embed",
}

# Families with a visible reasoning / thinking channel that HFL can
# expose via ``think=true``.
_THINKING_FAMILIES = {
    "gemma-4",
    "gemma4",
    "deepseek-r1",
    "deepseek_r1",
    "qwen3-thinking",
    "qwen3_thinking",
    "qwen-thinking",
    "gpt-oss",
    "gpt_oss",
    "o1",
    "o3",
    "reasoning",
}


def _matches_any(haystack: str, needles: Iterable[str]) -> bool:
    """Case-insensitive substring match against a set of needles."""
    hay = haystack.lower()
    return any(needle.lower() in hay for needle in needles)


def _search_corpus(manifest: "ModelManifest") -> str:
    """Concatenation of identity fields we match capabilities against.

    We search both ``name`` (human-readable alias) and ``repo_id``
    (canonical) plus the explicit architecture because naming
    conventions on HuggingFace are wildly inconsistent — a single
    source would miss half the matches.
    """
    parts: list[str] = []
    if manifest.name:
        parts.append(manifest.name)
    if manifest.repo_id:
        parts.append(manifest.repo_id)
    if manifest.architecture:
        parts.append(manifest.architecture)
    return " ".join(parts)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def detect_capabilities(manifest: "ModelManifest") -> list[str]:
    """Return the list of Ollama-style capabilities for a manifest.

    The order is stable (alphabetical inside each tier) so repeated
    calls produce the same list — test fixtures and doc examples
    therefore pin reliably.

    Rules:

    - ``completion`` — every manifest whose ``model_type`` is not
      explicitly ``"tts"`` / ``"stt"`` is assumed to support
      text completion.
    - ``tools`` — the family matcher in ``tool_parsers._detect_family``
      recognises the model.
    - ``insert`` — the family belongs to a FIM-capable set.
    - ``vision`` — the family is VL-known.
    - ``embedding`` — the architecture is an embedding family, OR
      the manifest's ``model_type`` is explicitly ``"embed"``.
    - ``thinking`` — the family is known to expose a reasoning channel.

    A model can carry multiple capabilities (e.g., Gemma 3 is both
    ``vision`` and ``tools``; Gemma 4 adds ``thinking``).
    """
    caps: list[str] = []
    corpus = _search_corpus(manifest)
    model_type = (manifest.model_type or "").lower()

    # Completion — every non-audio model
    if model_type not in {"tts", "stt", "audio"}:
        caps.append("completion")

    # Embedding — dedicated path (these models don't do completion)
    if model_type == "embed" or _matches_any(corpus, _EMBEDDING_FAMILIES):
        # Pure embedding models shouldn't also advertise completion —
        # drop it if we accidentally added it above.
        if "completion" in caps:
            caps.remove("completion")
        caps.append("embedding")

    # Tools
    if _matches_any(corpus, _TOOL_CAPABLE_FAMILIES):
        caps.append("tools")

    # Insert / FIM
    if _matches_any(corpus, _INSERT_CAPABLE_FAMILIES):
        caps.append("insert")

    # Vision
    if _matches_any(corpus, _VISION_FAMILIES):
        caps.append("vision")

    # Thinking / chain-of-thought
    if _matches_any(corpus, _THINKING_FAMILIES):
        caps.append("thinking")

    # Deterministic order: completion/embedding first (one of them),
    # then the rest alphabetical.
    primary = [c for c in caps if c in {"completion", "embedding"}]
    secondary = sorted(c for c in caps if c not in {"completion", "embedding"})
    return primary + secondary
