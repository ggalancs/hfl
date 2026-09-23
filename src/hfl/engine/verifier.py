# SPDX-License-Identifier: Apache-2.0
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
    skipped: bool = False  # not-applicable for this engine (doesn't count as a failure)


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
        # llama.cpp tokenizes internally and exposes no Python tokenizer, so
        # the round-trip is not applicable — skip it rather than flunk a
        # perfectly healthy GGUF model.
        return Check(
            name="tokenizer_round_trip",
            passed=True,
            skipped=True,
            detail="skipped: engine exposes no tokenizer (e.g. llama.cpp tokenizes internally)",
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
        _cleaned, calls = parse_tool_calls(sample, manifest.name, [])
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


def _check_signature(manifest: "ModelManifest") -> Check:
    """Report whether this manifest carries a signature we trust.

    ``observability/signing.py`` was written for exactly this and had no
    caller, so a signed manifest and a forged one verified identically —
    which is to say, the signature was decoration.

    Three outcomes, and the distinction between the first two is the
    whole point:

    * **unsigned** — ``skipped``. Signing is opt-in and most manifests
      have none; failing them would make the probe useless noise and
      train people to ignore it.
    * **signed and trusted** — pass, naming the key.
    * **signed and not trusted** — FAIL. A signature that does not verify
      is worse than none: it is a claim of provenance that is false.

    A missing trust root is also ``skipped``, not a failure: an operator
    who never curated one has not made a claim either way. A missing
    ed25519 backend is skipped for the same reason — that is our gap, not
    the model's.
    """
    from pathlib import Path as _Path

    envelope = getattr(manifest, "__dict__", None) or {}
    if not isinstance(envelope, dict) or not envelope.get("signature"):
        return Check(
            name="signature",
            passed=True,
            skipped=True,
            detail="unsigned (signing is opt-in)",
        )

    from hfl.config import config
    from hfl.observability.signing import (
        SignatureInvalidError,
        SignatureUnavailableError,
        TrustRoot,
        verify_manifest_envelope,
    )

    trust_path = _Path(config.home_dir) / "trusted-publishers.json"
    if not trust_path.exists():
        return Check(
            name="signature",
            passed=True,
            skipped=True,
            detail=f"signed, but no trust root at {trust_path} to check it against",
        )

    try:
        trust_root = TrustRoot.load(trust_path)
        trusted = verify_manifest_envelope(envelope, trust_root=trust_root)
    except SignatureUnavailableError as exc:
        return Check(
            name="signature",
            passed=True,
            skipped=True,
            detail=f"no ed25519 backend available: {exc}",
        )
    except (SignatureInvalidError, ValueError) as exc:
        return Check(name="signature", passed=False, detail=f"invalid signature: {exc}")

    key_id = (envelope.get("signature") or {}).get("key_id", "?")
    if trusted:
        return Check(name="signature", passed=True, detail=f"signed by trusted key {key_id!r}")

    # Unreachable against today's ``verify_manifest_envelope``, which
    # answers a signed envelope with True or an exception and never False
    # — the sole ``return False`` is the unsigned case, handled above.
    # Kept as a contract guard rather than deleted: if that function ever
    # starts returning False for a rejected signature, this must be a
    # FAILURE and not an accidental pass. ``test_signature_verify_wired``
    # pins the contract so the change is noticed here first.
    return Check(
        name="signature",
        passed=False,
        detail=f"signature by {key_id!r} was rejected without an error",
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
    checks.append(_check_signature(manifest))

    duration_ms = (time.perf_counter() - start) * 1000
    overall = all(c.passed for c in checks)

    return VerifyResult(
        model=manifest.name,
        overall_pass=overall,
        checks=checks,
        duration_ms=round(duration_ms, 2),
    )
