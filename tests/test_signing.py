# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ed25519 manifest signing (Phase 17 — V2 row 38)."""

from __future__ import annotations

import base64
import json
import os

import pytest

from hfl.observability import signing


def _fresh_keypair() -> tuple[bytes, bytes]:
    """Produce a seed + public key without committing to a particular backend."""
    try:
        from nacl.signing import SigningKey  # type: ignore

        sk = SigningKey(os.urandom(32))
        return bytes(sk), bytes(sk.verify_key)
    except ImportError:
        pass
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # type: ignore
        Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives.serialization import (  # type: ignore
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    priv = Ed25519PrivateKey.generate()
    private_bytes = priv.private_bytes(
        encoding=Encoding.Raw,
        format=PrivateFormat.Raw,
        encryption_algorithm=NoEncryption(),
    )
    public_bytes = priv.public_key().public_bytes(
        encoding=Encoding.Raw,
        format=PublicFormat.Raw,
    )
    return private_bytes, public_bytes


_SAMPLE = {
    "name": "llama3.3-q4",
    "repo_id": "meta-llama/Llama-3.3-70B",
    "file_hash": "abc123",
    "hash_algorithm": "sha256",
    "size_bytes": 1_234_567,
    "quantization": "Q4_K_M",
    "architecture": "llama",
}


class TestDigest:
    def test_stable_across_field_order(self):
        a = dict(_SAMPLE)
        b = {k: _SAMPLE[k] for k in reversed(list(_SAMPLE.keys()))}
        assert signing.manifest_digest(a) == signing.manifest_digest(b)

    def test_ignores_mutable_metadata(self):
        a = dict(_SAMPLE)
        a["last_used"] = "2026-04-17T00:00:00Z"
        assert signing.manifest_digest(a) == signing.manifest_digest(_SAMPLE)

    def test_changes_on_payload_mutation(self):
        a = dict(_SAMPLE)
        a["file_hash"] = "zzz"
        assert signing.manifest_digest(a) != signing.manifest_digest(_SAMPLE)


class TestSignVerifyRoundtrip:
    def test_happy_path(self):
        priv, pub = _fresh_keypair()
        signed = signing.sign_manifest_envelope(_SAMPLE, private_key=priv, key_id="tester")
        trust = signing.TrustRoot(
            keys={"tester": base64.urlsafe_b64encode(pub).decode().rstrip("=")}
        )
        assert signing.verify_manifest_envelope(signed, trust_root=trust) is True

    def test_verification_rejects_tampered_payload(self):
        priv, pub = _fresh_keypair()
        signed = signing.sign_manifest_envelope(_SAMPLE, private_key=priv, key_id="tester")
        tampered = dict(signed)
        tampered["file_hash"] = "deadbeef"
        trust = signing.TrustRoot(
            keys={"tester": base64.urlsafe_b64encode(pub).decode().rstrip("=")}
        )
        with pytest.raises(signing.SignatureInvalidError):
            signing.verify_manifest_envelope(tampered, trust_root=trust)

    def test_unknown_key_id_rejected(self):
        priv, _ = _fresh_keypair()
        signed = signing.sign_manifest_envelope(_SAMPLE, private_key=priv, key_id="alice")
        trust = signing.TrustRoot(keys={})
        with pytest.raises(signing.SignatureInvalidError):
            signing.verify_manifest_envelope(signed, trust_root=trust)

    def test_missing_signature_returns_false(self):
        trust = signing.TrustRoot(keys={})
        assert signing.verify_manifest_envelope(_SAMPLE, trust_root=trust) is False

    def test_malformed_signature_block_raises(self):
        signed = {**_SAMPLE, "signature": "not-an-object"}
        trust = signing.TrustRoot(keys={})
        with pytest.raises(signing.SignatureInvalidError):
            signing.verify_manifest_envelope(signed, trust_root=trust)

    def test_unsupported_algorithm_raises(self):
        signed = {
            **_SAMPLE,
            "signature": {"alg": "rsa", "key_id": "x", "digest": "y", "sig": "z"},
        }
        trust = signing.TrustRoot(keys={"x": "QUFB"})
        with pytest.raises(signing.SignatureInvalidError):
            signing.verify_manifest_envelope(signed, trust_root=trust)


class TestTrustRootLoad:
    def test_load_from_json(self, tmp_path):
        path = tmp_path / "root.json"
        path.write_text(json.dumps({"keys": {"k": "abc"}}))
        tr = signing.TrustRoot.load(path)
        assert tr.keys == {"k": "abc"}

    def test_load_rejects_malformed(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"not_keys": 1}')
        with pytest.raises(ValueError):
            signing.TrustRoot.load(path)
