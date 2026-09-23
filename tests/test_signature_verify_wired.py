# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Model signatures, finally checked by something.

`observability/signing.py` implemented ed25519 signing and verification
over a canonical manifest envelope, and said in its own docstring that
"`hfl verify` simply reports 'unsigned'". `hfl verify` did no such thing:
nothing imported the module, so a signed manifest and a forged one
verified identically. A signature nobody checks is decoration.

`verify_model` now runs a sixth probe. The three outcomes it
distinguishes are the substance:

* **unsigned** -> skipped. Signing is opt-in and most manifests carry
  none; failing them would make the probe noise, and a noisy probe is one
  people learn to ignore.
* **signed and trusted** -> pass.
* **signed and untrusted** -> FAIL. A signature that does not verify is
  worse than no signature: it is a false claim of provenance.

The tests below spend most of their length on that third case, because
it is the one where being wrong costs something.
"""

from __future__ import annotations

import base64
import json

import pytest

from hfl.engine.verifier import _check_signature
from hfl.observability.signing import manifest_digest, sign_manifest_envelope


class _Manifest:
    """Stand-in carrying only what the probe reads."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


BASE = {
    "name": "qwen-7b",
    "repo_id": "Qwen/Qwen2.5-7B",
    "file_hash": "abc123",
    "hash_algorithm": "sha256",
    "size_bytes": 4_000_000_000,
    "quantization": "Q4_K_M",
    "architecture": "qwen",
    "adapter_paths": [],
    "parent_digest": None,
}


@pytest.fixture
def hfl_home(tmp_path, monkeypatch):
    import hfl.config as hfl_config

    monkeypatch.setattr(hfl_config.config, "home_dir", tmp_path)
    return tmp_path


def _keypair():
    """Mint an ed25519 pair with whichever backend the module itself uses.

    ``signing.py`` tries pynacl and falls back to ``cryptography``; a
    fixture that only knew about pynacl skipped every meaningful case on
    a machine that has the other one — including "an untrusted signature
    fails", which is the assertion the whole feature rests on.

    Neither backend is present in the CI venv (no ``[llama]``, no
    ``[otel]``, and ``cryptography`` arrives only as a transitive dep of
    the dev extras), so these cases run on a developer machine and skip
    in CI. The plumbing cases below need no backend and run everywhere —
    that split is deliberate, not an oversight.
    """
    try:
        from nacl.signing import SigningKey

        signing_key = SigningKey.generate()
        return bytes(signing_key), bytes(signing_key.verify_key)
    except ImportError:
        pass
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError:  # pragma: no cover — depends on the venv
        pytest.skip("no ed25519 backend (pynacl or cryptography) installed")

    private = Ed25519PrivateKey.generate()
    raw_private = private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    raw_public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return raw_private, raw_public


def _trust_file(home, key_id, public):
    path = home / "trusted-publishers.json"
    path.write_text(
        json.dumps({"keys": {key_id: base64.urlsafe_b64encode(public).decode().rstrip("=")}}),
        encoding="utf-8",
    )
    return path


class TestUnsignedIsNotAFailure:
    def test_an_unsigned_manifest_is_skipped(self, hfl_home):
        check = _check_signature(_Manifest(**BASE))
        assert check.skipped is True
        assert check.passed is True
        assert "unsigned" in check.detail

    def test_it_does_not_drag_down_the_overall_result(self, hfl_home):
        """A skipped check must not turn every unsigned model into a failure."""
        check = _check_signature(_Manifest(**BASE))
        assert check.passed, (
            "unsigned manifests would now fail `hfl verify`, which is most of "
            "them — the probe would be pure noise"
        )


class TestATrustedSignaturePasses:
    def test_signed_by_a_known_key(self, hfl_home):
        private, public = _keypair()
        _trust_file(hfl_home, "publisher-a", public)
        signed = sign_manifest_envelope(dict(BASE), private_key=private, key_id="publisher-a")

        check = _check_signature(_Manifest(**signed))
        assert check.passed and not check.skipped
        assert "publisher-a" in check.detail


class TestAnUntrustedSignatureFails:
    """The case that justifies the feature."""

    def test_a_key_we_do_not_trust_is_a_failure(self, hfl_home):
        private, _public = _keypair()
        _other_private, other_public = _keypair()
        # The trust root holds somebody else's key.
        _trust_file(hfl_home, "publisher-b", other_public)
        signed = sign_manifest_envelope(dict(BASE), private_key=private, key_id="publisher-a")

        check = _check_signature(_Manifest(**signed))
        assert check.passed is False, (
            "a manifest signed by an unknown key verified as fine, which is the "
            "entire failure this check exists to catch"
        )
        assert not check.skipped

    def test_a_known_key_with_bad_signature_bytes_fails(self, hfl_home):
        """Right key id, wrong key behind it."""
        private, _public = _keypair()
        _other_private, other_public = _keypair()
        _trust_file(hfl_home, "publisher-a", other_public)
        signed = sign_manifest_envelope(dict(BASE), private_key=private, key_id="publisher-a")

        check = _check_signature(_Manifest(**signed))
        assert check.passed is False, (
            "the signature did not verify against the trusted key and the probe "
            "passed anyway — a false provenance claim accepted as genuine"
        )
        assert not check.skipped

    def test_a_tampered_manifest_fails(self, hfl_home):
        """Signed, trusted key, but the contents changed afterwards."""
        private, public = _keypair()
        _trust_file(hfl_home, "publisher-a", public)
        signed = sign_manifest_envelope(dict(BASE), private_key=private, key_id="publisher-a")

        # Swap the blob digest the signature pins.
        signed["file_hash"] = "deadbeef"
        assert manifest_digest(signed) != signed["signature"]["digest"]

        check = _check_signature(_Manifest(**signed))
        assert check.passed is False

    def test_a_malformed_signature_block_fails(self, hfl_home):
        _private, public = _keypair()
        _trust_file(hfl_home, "publisher-a", public)
        broken = dict(BASE)
        broken["signature"] = {"alg": "rsa", "key_id": "x", "digest": "y", "sig": "z"}

        check = _check_signature(_Manifest(**broken))
        assert check.passed is False


class TestTheContractTheProbeRelieson:
    """`verify_manifest_envelope` answers True-or-raise, never False.

    Found by sabotage: the probe originally ended with an ``if trusted:
    ... else: fail`` pair, and rewriting the failure branch to pass
    changed nothing, because no signed envelope ever reaches it — every
    rejection arrives as an exception. That branch was unreachable code
    pretending to be a safeguard.

    It is kept as a contract guard, so these tests pin the contract. If
    the function ever starts returning False for a rejected signature,
    they fail here rather than letting the probe quietly pass everything.
    """

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(
                lambda e: e.update({"signature": {**e["signature"], "key_id": "nobody"}}),
                id="unknown-key",
            ),
            pytest.param(lambda e: e.update({"file_hash": "tampered"}), id="tampered"),
            pytest.param(
                lambda e: e.update({"signature": {**e["signature"], "sig": "AAAA"}}), id="bad-sig"
            ),
        ],
    )
    def test_every_rejection_raises_rather_than_returning_false(self, hfl_home, mutate):
        from hfl.observability.signing import (
            SignatureInvalidError,
            TrustRoot,
            verify_manifest_envelope,
        )

        private, public = _keypair()
        path = _trust_file(hfl_home, "publisher-a", public)
        signed = sign_manifest_envelope(dict(BASE), private_key=private, key_id="publisher-a")
        mutate(signed)

        with pytest.raises(SignatureInvalidError):
            verify_manifest_envelope(signed, trust_root=TrustRoot.load(path))

    def test_a_good_signature_returns_true(self, hfl_home):
        from hfl.observability.signing import TrustRoot, verify_manifest_envelope

        private, public = _keypair()
        path = _trust_file(hfl_home, "publisher-a", public)
        signed = sign_manifest_envelope(dict(BASE), private_key=private, key_id="publisher-a")
        assert verify_manifest_envelope(signed, trust_root=TrustRoot.load(path)) is True


class TestMissingTrustRootIsNotAVerdict:
    def test_no_keyring_means_skipped_not_failed(self, hfl_home):
        """An operator who never curated a trust root has made no claim.

        Failing here would punish them for the feature existing.
        """
        private, _public = _keypair()
        signed = sign_manifest_envelope(dict(BASE), private_key=private, key_id="publisher-a")
        assert not (hfl_home / "trusted-publishers.json").exists()

        check = _check_signature(_Manifest(**signed))
        assert check.skipped is True
        assert check.passed is True
        assert "trust root" in check.detail


class TestTheProbeIsActuallyRun:
    def test_verify_model_includes_it(self):
        import ast
        import inspect
        import textwrap

        from hfl.engine import verifier

        tree = ast.parse(textwrap.dedent(inspect.getsource(verifier.verify_model)))
        called = {getattr(n.func, "id", None) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        assert "_check_signature" in called, (
            "verify_model does not run the signature probe, so it is defined "
            "and never executed — the state this whole change was about"
        )
