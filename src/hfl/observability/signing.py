# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""ed25519 model-signature plumbing (Phase 17 — V2 row 38).

Manifests can carry an ed25519 signature over a canonical envelope
that pins every blob digest that makes up the model. Verification
checks the signature against a trust-root keyring the operator
curates under ``~/.hfl/trusted-publishers.json``.

The module is self-contained — no hfl.config imports — so
verification is cheap to run from the CLI without bringing up the
full server. Signing is opt-in: manifests without ``signature``
stay valid today (we don't break existing users) and ``hfl verify``
simply reports "unsigned".

Stdlib has no ed25519 primitive; we depend on ``pynacl`` or
``cryptography``, whichever is importable. Both are already
transitive dependencies of pieces we ship (``huggingface_hub`` pulls
``cryptography``).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "sign_manifest_envelope",
    "verify_manifest_envelope",
    "SignatureUnavailableError",
    "SignatureInvalidError",
    "TrustRoot",
    "manifest_digest",
]


class SignatureUnavailableError(RuntimeError):
    """Raised when no ed25519 primitive is importable."""


class SignatureInvalidError(ValueError):
    """Raised when a signature fails verification."""


@dataclass
class TrustRoot:
    """In-memory keyring the server consults for verifications."""

    # Map ``key_id -> public_key_base64_url``. Key ids are
    # typically short human names (``github/ggalancs``); the public
    # key is a 32-byte ed25519 raw key, base64url-encoded.
    keys: dict[str, str]

    @classmethod
    def load(cls, path: str | Path) -> "TrustRoot":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        keys = data.get("keys") if isinstance(data, dict) else None
        if not isinstance(keys, dict):
            raise ValueError(f"trust root {path} has no 'keys' mapping")
        return cls(keys={str(k): str(v) for k, v in keys.items()})

    def public_key(self, key_id: str) -> bytes | None:
        raw = self.keys.get(key_id)
        if raw is None:
            return None
        pad = "=" * (-len(raw) % 4)
        return base64.urlsafe_b64decode(raw + pad)


# ----------------------------------------------------------------------
# Canonical envelope
# ----------------------------------------------------------------------


def manifest_digest(envelope: dict[str, Any]) -> str:
    """Compute a stable SHA-256 digest of the manifest envelope.

    Only the fields that matter for provenance get included — we
    intentionally drop mutable metadata (``last_used``, timestamps)
    so a plain ``hfl show`` or a ``verified_at`` update doesn't
    invalidate the signature.
    """
    carved: dict[str, Any] = {
        "name": envelope.get("name"),
        "repo_id": envelope.get("repo_id"),
        "file_hash": envelope.get("file_hash"),
        "hash_algorithm": envelope.get("hash_algorithm"),
        "size_bytes": envelope.get("size_bytes"),
        "quantization": envelope.get("quantization"),
        "architecture": envelope.get("architecture"),
        "adapter_paths": envelope.get("adapter_paths", []),
        "parent_digest": envelope.get("parent_digest"),
    }
    text = json.dumps(carved, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# Backend selection
# ----------------------------------------------------------------------


def _sign_raw(private_key: bytes, message: bytes) -> bytes:
    try:
        from nacl.signing import SigningKey  # type: ignore

        return bytes(SigningKey(private_key).sign(message).signature)
    except ImportError:
        pass
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # type: ignore
            Ed25519PrivateKey,
        )
    except ImportError as exc:
        raise SignatureUnavailableError(
            "No ed25519 backend installed (need pynacl or cryptography)"
        ) from exc
    key = Ed25519PrivateKey.from_private_bytes(private_key)
    return key.sign(message)


def _verify_raw(public_key: bytes, signature: bytes, message: bytes) -> bool:
    try:
        from nacl.signing import VerifyKey  # type: ignore

        try:
            VerifyKey(public_key).verify(message, signature)
            return True
        except Exception:
            return False
    except ImportError:
        pass
    try:
        from cryptography.exceptions import InvalidSignature  # type: ignore
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # type: ignore
            Ed25519PublicKey,
        )
    except ImportError as exc:
        raise SignatureUnavailableError(
            "No ed25519 backend installed (need pynacl or cryptography)"
        ) from exc
    key = Ed25519PublicKey.from_public_bytes(public_key)
    try:
        key.verify(signature, message)
        return True
    except InvalidSignature:
        return False


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def sign_manifest_envelope(
    envelope: dict[str, Any],
    *,
    private_key: bytes,
    key_id: str,
) -> dict[str, Any]:
    """Return a copy of ``envelope`` with a ``signature`` block attached."""
    digest = manifest_digest(envelope)
    sig = _sign_raw(private_key, digest.encode("ascii"))
    out = dict(envelope)
    out["signature"] = {
        "alg": "ed25519",
        "key_id": key_id,
        "digest": digest,
        "sig": base64.urlsafe_b64encode(sig).decode("ascii"),
    }
    return out


def verify_manifest_envelope(
    envelope: dict[str, Any],
    *,
    trust_root: TrustRoot,
) -> bool:
    """Return True iff the manifest carries a valid signature we trust.

    Raises ``SignatureInvalidError`` when the signature block is
    present but malformed / unauthorized — the operator set strict
    mode and wants a 400 on bad envelopes.
    """
    sig_block = envelope.get("signature")
    if not sig_block:
        return False
    if not isinstance(sig_block, dict):
        raise SignatureInvalidError("signature block must be an object")
    if sig_block.get("alg") != "ed25519":
        raise SignatureInvalidError(f"unsupported signature algorithm: {sig_block.get('alg')!r}")
    key_id = sig_block.get("key_id")
    if not isinstance(key_id, str):
        raise SignatureInvalidError("signature.key_id missing or non-string")
    pub = trust_root.public_key(key_id)
    if pub is None:
        raise SignatureInvalidError(f"signature.key_id {key_id!r} not in trust root")
    expected_digest = manifest_digest(envelope)
    if sig_block.get("digest") != expected_digest:
        raise SignatureInvalidError("signed digest does not match manifest contents")
    raw_sig = sig_block.get("sig")
    if not isinstance(raw_sig, str):
        raise SignatureInvalidError("signature.sig missing or non-string")
    try:
        sig = base64.urlsafe_b64decode(raw_sig + "=" * (-len(raw_sig) % 4))
    except Exception as exc:
        raise SignatureInvalidError("signature.sig is not valid base64url") from exc
    ok = _verify_raw(pub, sig, expected_digest.encode("ascii"))
    if not ok:
        raise SignatureInvalidError("ed25519 verification failed")
    return True
