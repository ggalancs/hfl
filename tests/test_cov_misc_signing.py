# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Manifest signatures with either ed25519 backend, or none.

Neither pynacl nor cryptography need be installed: each backend is a small
fake in ``sys.modules`` (a keyed hash, not ed25519), enough to drive HFL's
own plumbing — which backend is used, what is signed, and every way a
signature block is refused."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import types

import pytest

from hfl.observability import signing
from hfl.observability.signing import (
    SignatureInvalidError,
    SignatureUnavailableError,
    TrustRoot,
    manifest_digest,
    sign_manifest_envelope,
    verify_manifest_envelope,
)

SEED = b"s" * 32
ENVELOPE = {
    "name": "q05",
    "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "file_hash": "abc",
    "hash_algorithm": "sha256",
    "size_bytes": 10,
    "quantization": "Q4_K_M",
    "architecture": "qwen2",
    "last_used": "2026-01-01",
}


def _public(seed: bytes) -> bytes:
    return hashlib.sha256(b"pub" + seed).digest()


def _sig(public: bytes, message: bytes) -> bytes:
    return hashlib.sha512(public + message).digest()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


class _BadSignature(Exception):
    pass


def _fake_nacl(used: list[str]) -> dict[str, types.ModuleType]:
    nacl = types.ModuleType("nacl")
    mod = types.ModuleType("nacl.signing")

    class SigningKey:
        def __init__(self, seed: bytes) -> None:
            self.public = _public(seed)

        def sign(self, message: bytes):
            used.append("nacl.sign")
            return types.SimpleNamespace(signature=_sig(self.public, message))

    class VerifyKey:
        def __init__(self, public: bytes) -> None:
            self.public = public

        def verify(self, message: bytes, signature: bytes) -> bytes:
            used.append("nacl.verify")
            if _sig(self.public, message) != signature:
                raise _BadSignature("bad")
            return message

    mod.SigningKey = SigningKey  # type: ignore[attr-defined]
    mod.VerifyKey = VerifyKey  # type: ignore[attr-defined]
    nacl.signing = mod  # type: ignore[attr-defined]
    return {"nacl": nacl, "nacl.signing": mod}


def _fake_cryptography(used: list[str]) -> dict[str, types.ModuleType]:
    names = [
        "cryptography",
        "cryptography.exceptions",
        "cryptography.hazmat",
        "cryptography.hazmat.primitives",
        "cryptography.hazmat.primitives.asymmetric",
        "cryptography.hazmat.primitives.asymmetric.ed25519",
    ]
    mods = {n: types.ModuleType(n) for n in names}

    class InvalidSignature(Exception):
        pass

    class Ed25519PrivateKey:
        def __init__(self, seed: bytes) -> None:
            self.public = _public(seed)

        @classmethod
        def from_private_bytes(cls, seed: bytes) -> Ed25519PrivateKey:
            return cls(seed)

        def sign(self, message: bytes) -> bytes:
            used.append("crypto.sign")
            return _sig(self.public, message)

    class Ed25519PublicKey:
        def __init__(self, public: bytes) -> None:
            self.public = public

        @classmethod
        def from_public_bytes(cls, public: bytes) -> Ed25519PublicKey:
            return cls(public)

        def verify(self, signature: bytes, message: bytes) -> None:
            used.append("crypto.verify")
            if _sig(self.public, message) != signature:
                raise InvalidSignature()

    mods["cryptography.exceptions"].InvalidSignature = InvalidSignature  # type: ignore[attr-defined]
    ed = mods["cryptography.hazmat.primitives.asymmetric.ed25519"]
    ed.Ed25519PrivateKey = Ed25519PrivateKey  # type: ignore[attr-defined]
    ed.Ed25519PublicKey = Ed25519PublicKey  # type: ignore[attr-defined]
    return mods


@pytest.fixture
def backend(monkeypatch):
    """Selects a backend: "nacl", "cryptography" or "none". Returns the log
    of which fake primitive ran."""
    used: list[str] = []

    def select(which: str) -> list[str]:
        nacl = _fake_nacl(used)
        crypto = _fake_cryptography(used)
        for name, mod in {**nacl, **crypto}.items():
            keep = (which == "nacl" and name in nacl) or (
                which == "cryptography" and name in crypto
            )
            # None in sys.modules makes the import raise ImportError.
            monkeypatch.setitem(sys.modules, name, mod if keep else None)
        return used

    return select


def _trust(key_id: str = "pub/one") -> TrustRoot:
    return TrustRoot(keys={key_id: _b64(_public(SEED))})


# -- the trust root ----------------------------------------------------------------


def test_trust_root_decodes_unpadded_base64url_keys() -> None:
    root = _trust()
    assert root.public_key("pub/one") == _public(SEED)
    assert root.public_key("someone-else") is None


def test_trust_root_loads_from_a_file(tmp_path) -> None:
    path = tmp_path / "trusted-publishers.json"
    path.write_text(json.dumps({"keys": {"a": "AAAA", 7: 8}}))
    assert TrustRoot.load(path).keys == {"a": "AAAA", "7": "8"}


@pytest.mark.parametrize("content", [[1, 2], {"keys": ["a"]}, {"nokeys": {}}])
def test_trust_root_without_a_keys_mapping_is_refused(tmp_path, content) -> None:
    path = tmp_path / "t.json"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="no 'keys' mapping"):
        TrustRoot.load(path)


# -- the digest -----------------------------------------------------------------------


def test_the_digest_ignores_mutable_metadata_but_not_provenance() -> None:
    base = manifest_digest(ENVELOPE)
    assert manifest_digest({**ENVELOPE, "last_used": "later", "verified_at": "x"}) == base
    assert manifest_digest({**ENVELOPE, "file_hash": "other"}) != base
    assert manifest_digest({**ENVELOPE, "adapter_paths": ["/a"]}) != base
    assert len(base) == 64


# -- signing and verifying, per backend ----------------------------------------------------


@pytest.mark.parametrize("which", ["nacl", "cryptography"])
def test_a_signed_envelope_verifies(backend, which) -> None:
    used = backend(which)
    signed = sign_manifest_envelope(ENVELOPE, private_key=SEED, key_id="pub/one")
    block = signed["signature"]
    assert (block["alg"], block["key_id"]) == ("ed25519", "pub/one")
    assert block["digest"] == manifest_digest(ENVELOPE)
    assert "signature" not in ENVELOPE  # a copy, the input untouched
    assert verify_manifest_envelope(signed, trust_root=_trust()) is True
    prefix = "nacl" if which == "nacl" else "crypto"
    assert used == [f"{prefix}.sign", f"{prefix}.verify"]


@pytest.mark.parametrize("which", ["nacl", "cryptography"])
def test_a_forged_signature_fails(backend, which) -> None:
    backend(which)
    signed = sign_manifest_envelope(ENVELOPE, private_key=SEED, key_id="pub/one")
    signed["signature"]["sig"] = _b64(b"\x00" * 64)
    with pytest.raises(SignatureInvalidError, match="verification failed"):
        verify_manifest_envelope(signed, trust_root=_trust())


@pytest.mark.parametrize("which", ["nacl", "cryptography"])
def test_a_key_not_matching_the_trust_root_fails(backend, which) -> None:
    backend(which)
    signed = sign_manifest_envelope(ENVELOPE, private_key=b"x" * 32, key_id="pub/one")
    with pytest.raises(SignatureInvalidError, match="verification failed"):
        verify_manifest_envelope(signed, trust_root=_trust())


def test_without_a_backend_signing_says_what_to_install(backend) -> None:
    backend("none")
    with pytest.raises(SignatureUnavailableError, match="pynacl or cryptography"):
        sign_manifest_envelope(ENVELOPE, private_key=SEED, key_id="pub/one")


def test_without_a_backend_verifying_says_what_to_install(backend) -> None:
    backend("none")
    digest = manifest_digest(ENVELOPE)
    envelope = {
        **ENVELOPE,
        "signature": {"alg": "ed25519", "key_id": "pub/one", "digest": digest, "sig": "AAAA"},
    }
    with pytest.raises(SignatureUnavailableError, match="pynacl or cryptography"):
        verify_manifest_envelope(envelope, trust_root=_trust())


# -- malformed blocks are refused, never verified -------------------------------------------


def test_an_unsigned_envelope_is_simply_unsigned() -> None:
    assert verify_manifest_envelope(dict(ENVELOPE), trust_root=_trust()) is False
    assert verify_manifest_envelope({**ENVELOPE, "signature": {}}, trust_root=_trust()) is False


def _block(**overrides) -> dict:
    block = {
        "alg": "ed25519",
        "key_id": "pub/one",
        "digest": manifest_digest(ENVELOPE),
        "sig": _b64(b"\x01" * 64),
    }
    block.update(overrides)
    return {**ENVELOPE, "signature": block}


@pytest.mark.parametrize(
    ("envelope", "message"),
    [
        ({**ENVELOPE, "signature": "not-a-dict"}, "must be an object"),
        (_block(alg="rsa"), "unsupported signature algorithm: 'rsa'"),
        (_block(key_id=5), "key_id missing or non-string"),
        (_block(key_id="stranger"), "not in trust root"),
        (_block(digest="0" * 64), "does not match manifest contents"),
        (_block(sig=None), "sig missing or non-string"),
        (_block(sig="A"), "not valid base64url"),
    ],
)
def test_malformed_signature_blocks_are_refused(backend, envelope, message) -> None:
    used = backend("nacl")
    with pytest.raises(SignatureInvalidError, match=message):
        verify_manifest_envelope(envelope, trust_root=_trust())
    assert used == []  # refused before any cryptography ran


def test_a_tampered_manifest_no_longer_matches_its_digest(backend) -> None:
    backend("nacl")
    signed = sign_manifest_envelope(ENVELOPE, private_key=SEED, key_id="pub/one")
    signed["file_hash"] = "swapped-blob"
    with pytest.raises(SignatureInvalidError, match="digest does not match"):
        verify_manifest_envelope(signed, trust_root=_trust())


def test_exports() -> None:
    assert set(signing.__all__) >= {"sign_manifest_envelope", "verify_manifest_envelope"}
