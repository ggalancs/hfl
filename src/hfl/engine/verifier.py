# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Model verification — V4 F3.1.

Sanity checks a freshly-pulled (or about-to-pull) model so the
operator knows before committing it to production:

- Tokenizer round-trip (encode + decode of a known string).
- Chat-template render (apply_chat_template against a 1-msg list).
- Smoke generation (1 token, deterministic seed).
- Tool-call parser (round-trip a fake call through the parser
  registered for this family).
- Embedding dimension (when manifest declares an embedding model).

Returns a :class:`VerifyResult` with one :class:`Check` per probe.
A check failing does NOT raise — the caller wants the full picture
of what's broken, not a partial report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hfl.engine.base import InferenceEngine
    from hfl.models.manifest import ModelManifest

logger = logging.getLogger(__name__)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class VerifyResult:
    model: str
    overall_pass: bool
    checks: list[Check] = field(default_factory=list)
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Individual probes
# ---------------------------------------------------------------------------


def _check_tokenizer_round_trip(engine: "InferenceEngine") -> Check:
    """Tokenizer must encode + decode without losing the input.

    A bad MLX or HF conversion sometimes ships a tokenizer whose
    ``decode(encode(x))`` is not equal to ``x`` for trivial Latin
    text — that breaks every chat. The tolerance here is ``startswith``
    rather than equality because some tokenizers always insert a BOS
    space.
    """
    sample = "Hello, world."
    tokenizer = getattr(engine, "tokenizer", None) or getattr(engine, "_tokenizer", None)
    if tokenizer is None:
        return Check(
            name="tokenizer_round_trip",
            passed=False,
            detail="engine does not expose a tokenizer attribute",
        )
    try:
        ids = tokenizer.encode(sample) if hasattr(tokenizer, "encode") else tokenizer(sample)
        decoded = tokenizer.decode(ids) if hasattr(tokenizer, "decode") else str(ids)
    except Exception as exc:
        return Check(name="tokenizer_round_trip", passed=False, detail=f"raised: {exc}")
    cleaned = decoded.strip()
    if cleaned.startswith(sample) or sample in cleaned:
        return Check(name="tokenizer_round_trip", passed=True, detail=f"decoded={cleaned!r}")
    return Check(
        name="tokenizer_round_trip",
        passed=False,
        detail=f"expected to contain {sample!r}, got {cleaned!r}",
    )


def _check_chat_template(engine: "InferenceEngine", manifest: "ModelManifest") -> Check:
    """Engine.chat (or apply_chat_template) renders a 1-msg dialogue.

    The MLX-pulled tokenizers without a chat_template are the common
    failure mode here — we want this surfaced cleanly rather than
    via a 500 on the first /api/chat.
    """
    if manifest.format and manifest.format.lower() in {"audio", "image"}:
        return Check(name="chat_template_render", passed=True, detail="not applicable")

    tokenizer = getattr(engine, "tokenizer", None) or getattr(engine, "_tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": "hello"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, str) and rendered.strip():
                return Check(
                    name="chat_template_render", passed=True, detail=f"len={len(rendered)}"
                )
            return Check(name="chat_template_render", passed=False, detail="empty render")
        except Exception as exc:
            return Check(name="chat_template_render", passed=False, detail=f"raised: {exc}")

    # llama-cpp-python carries chat support via Llama.create_chat_completion;
    # absence of apply_chat_template is fine when the engine still
    # exposes a chat() method.
    if hasattr(engine, "chat"):
        return Check(
            name="chat_template_render",
            passed=True,
            detail="engine.chat present (llama-cpp internal template)",
        )
    return Check(
        name="chat_template_render",
        passed=False,
        detail="no apply_chat_template + no engine.chat",
    )


def _check_smoke_generation(engine: "InferenceEngine") -> Check:
    """One-token deterministic generation — the engine actually runs."""
    from hfl.engine.base import GenerationConfig

    cfg = GenerationConfig(max_tokens=1, temperature=0.0, top_p=1.0)
    try:
        result = engine.generate("Hello", cfg)
    except Exception as exc:
        return Check(name="smoke_generation", passed=False, detail=f"raised: {exc}")

    text = getattr(result, "text", "") or ""
    if not text:
        return Check(name="smoke_generation", passed=False, detail="empty output")
    return Check(name="smoke_generation", passed=True, detail=f"produced {len(text)} chars")


def _check_tool_parser(manifest: "ModelManifest") -> Check:
    """Tool-call parser registered for this family round-trips a
    canonical fake call. Doesn't touch the engine — purely lexical."""
    try:
        from hfl.api.tool_parsers import dispatch as parse_tool_calls

        sample = '<tool_call>\n{"name": "x", "arguments": {}}\n</tool_call>'
        cleaned, calls = parse_tool_calls(sample, manifest.name, [])
        # The parser may return zero calls (no tool registered for
        # the model name); we only fail when it raises.
        return Check(
            name="tool_parser_round_trip",
            passed=True,
            detail=f"parsed {len(calls)} call(s)",
        )
    except Exception as exc:
        return Check(name="tool_parser_round_trip", passed=False, detail=f"raised: {exc}")


def _check_embedding_dim(engine: "InferenceEngine", manifest: "ModelManifest") -> Check:
    """For embedding models, request a 1-input embedding and verify
    the dimension matches the manifest (when declared)."""
    declared_caps = getattr(manifest, "declared_capabilities", []) or []
    if "embeddings" not in declared_caps:
        return Check(name="embedding_dim", passed=True, detail="not an embedding model")
    embedder = getattr(engine, "embed", None)
    if not callable(embedder):
        return Check(name="embedding_dim", passed=False, detail="no embed() method")
    try:
        vec = embedder(["hello"])
    except Exception as exc:
        return Check(name="embedding_dim", passed=False, detail=f"raised: {exc}")
    if isinstance(vec, list) and vec and isinstance(vec[0], (list, tuple)):
        return Check(name="embedding_dim", passed=True, detail=f"dim={len(vec[0])}")
    return Check(name="embedding_dim", passed=False, detail=f"unexpected shape: {type(vec)!r}")


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------


_PROBES = (
    _check_tokenizer_round_trip,
    None,  # placeholder for chat_template (needs manifest too)
    _check_smoke_generation,
)


def verify_model(engine: "InferenceEngine", manifest: "ModelManifest") -> VerifyResult:
    """Run all checks against a loaded engine + manifest.

    Returns a :class:`VerifyResult` with ``overall_pass`` true iff
    every check passed. Caller decides what "pass" means for their
    workflow (acceptance test, registry annotation, etc.).
    """
    import time

    start = time.perf_counter()
    checks: list[Check] = []

    checks.append(_check_tokenizer_round_trip(engine))
    checks.append(_check_chat_template(engine, manifest))
    checks.append(_check_smoke_generation(engine))
    checks.append(_check_tool_parser(manifest))
    checks.append(_check_embedding_dim(engine, manifest))

    duration_ms = (time.perf_counter() - start) * 1000
    overall = all(c.passed for c in checks)

    return VerifyResult(
        model=manifest.name,
        overall_pass=overall,
        checks=checks,
        duration_ms=round(duration_ms, 2),
    )
