# SPDX-License-Identifier: HRUL-1.0
"""Security edge cases for ed25519 manifest signature verification.

Complements ``test_signing.py`` by driving the rejection paths in
``verify_manifest_envelope`` (a tampered or malformed signature block must
raise ``SignatureInvalidError``, never silently verify) and the
``cryptography`` fallback backend that the nacl-first path normally hides.
"""

from __future__ import annotations

import base64
import importlib.util
import os
import sys

import pytest

from hfl.observability.signing import (
    SignatureInvalidError,
    TrustRoot,
    sign_manifest_envelope,
    verify_manifest_envelope,
)


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except ModuleNotFoundError:
        return False


_HAS_CRYPTOGRAPHY = _has("cryptography")
# Signing/verifying needs SOME ed25519 backend. CI may ship neither (the
# existing test_signing.py skips wholesale in that case); match that so these
# tests skip cleanly instead of erroring when no backend is installed.
_HAS_BACKEND = _HAS_CRYPTOGRAPHY or _has("nacl")

_SAMPLE = {
    "name": "qwen3-coder",
    "repo_id": "Qwen/Qwen3-Coder-30B",
    "file_hash": "deadbeef",
    "hash_algorithm": "sha256",
    "size_bytes": 17_000_000,
    "quantization": "Q4_K_M",
    "architecture": "qwen",
}

_KEY_ID = "test-key-1"


def _keypair() -> tuple[bytes, bytes]:
    """Raw 32-byte ed25519 seed + public key, backend-agnostic.

    Tries pynacl first (the only backend guaranteed in CI), then falls back
    to cryptography. The raw seed/pubkey bytes work with either backend.
    """
    try:
        from nacl.signing import SigningKey  # type: ignore

        sk = SigningKey(os.urandom(32))
        return bytes(sk), bytes(sk.verify_key)
    except ImportError:
        pass
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    priv = Ed25519PrivateKey.generate()
    seed = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    pub = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return seed, pub


def _trust_root(pub: bytes, key_id: str = _KEY_ID) -> TrustRoot:
    encoded = base64.urlsafe_b64encode(pub).decode("ascii").rstrip("=")
    return TrustRoot(keys={key_id: encoded})


def _signed():
    seed, pub = _keypair()
    signed = sign_manifest_envelope(dict(_SAMPLE), private_key=seed, key_id=_KEY_ID)
    return signed, _trust_root(pub)


@pytest.mark.skipif(not _HAS_BACKEND, reason="no ed25519 backend installed")
class TestVerifyRejectionPaths:
    def test_key_id_non_string_rejected(self):
        env = dict(_SAMPLE)
        env["signature"] = {"alg": "ed25519", "key_id": 123, "digest": "x", "sig": "y"}
        with pytest.raises(SignatureInvalidError, match="key_id missing or non-string"):
            verify_manifest_envelope(env, trust_root=_trust_root(b"\x00" * 32))

    def test_sig_non_string_rejected(self):
        signed, root = _signed()
        signed["signature"]["sig"] = 123  # valid alg/key_id/digest, bad sig type
        with pytest.raises(SignatureInvalidError, match="sig missing or non-string"):
            verify_manifest_envelope(signed, trust_root=root)

    def test_sig_invalid_base64_rejected(self):
        signed, root = _signed()
        signed["signature"]["sig"] = "A"  # length 1 -> bad padding -> decode error
        with pytest.raises(SignatureInvalidError, match="not valid base64url"):
            verify_manifest_envelope(signed, trust_root=root)

    def test_wrong_signature_rejected(self):
        signed, root = _signed()
        # Valid base64url, correct digest, but the signature bytes are bogus.
        signed["signature"]["sig"] = base64.urlsafe_b64encode(b"\x00" * 64).decode("ascii")
        with pytest.raises(SignatureInvalidError, match="ed25519 verification failed"):
            verify_manifest_envelope(signed, trust_root=root)

    def test_valid_signature_accepted(self):
        signed, root = _signed()
        assert verify_manifest_envelope(signed, trust_root=root) is True


@pytest.mark.skipif(not _HAS_CRYPTOGRAPHY, reason="cryptography backend not installed")
class TestCryptographyFallback:
    """Force the nacl import to fail so the cryptography backend is exercised."""

    @pytest.fixture
    def no_nacl(self, monkeypatch):
        # Setting the submodule to None makes `from nacl.signing import ...`
        # raise ImportError, dropping _sign_raw/_verify_raw to cryptography.
        monkeypatch.setitem(sys.modules, "nacl", None)
        monkeypatch.setitem(sys.modules, "nacl.signing", None)
        yield

    def test_sign_and_verify_roundtrip_via_cryptography(self, no_nacl):
        seed, pub = _keypair()
        signed = sign_manifest_envelope(dict(_SAMPLE), private_key=seed, key_id=_KEY_ID)
        assert verify_manifest_envelope(signed, trust_root=_trust_root(pub)) is True

    def test_cryptography_rejects_tampered_signature(self, no_nacl):
        seed, pub = _keypair()
        signed = sign_manifest_envelope(dict(_SAMPLE), private_key=seed, key_id=_KEY_ID)
        signed["signature"]["sig"] = base64.urlsafe_b64encode(b"\x01" * 64).decode("ascii")
        with pytest.raises(SignatureInvalidError, match="ed25519 verification failed"):
            verify_manifest_envelope(signed, trust_root=_trust_root(pub))
