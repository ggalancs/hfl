# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Unit tests for the owner-vs-user trust boundary.

Covers :mod:`hfl.api.admin_guard` (who may trigger administrative
endpoints) and :func:`hfl.hub.license_checker.policy_allows` (which
license tiers the owner pre-accepts for non-interactive pulls).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from hfl.api.admin_guard import is_local_request, require_owner
from hfl.hub.license_checker import LicenseInfo, LicenseRisk, policy_allows


def _request(host: str | None):
    """A stand-in for starlette's Request carrying only ``.client.host``."""
    client = None if host is None else SimpleNamespace(host=host)
    return SimpleNamespace(client=client)


class TestIsLocalRequest:
    @pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
    def test_loopback_hosts_are_local(self, host):
        assert is_local_request(_request(host)) is True

    @pytest.mark.parametrize("host", ["203.0.113.7", "10.0.0.5", "192.168.1.20", "testclient"])
    def test_non_loopback_hosts_are_remote(self, host):
        assert is_local_request(_request(host)) is False

    def test_missing_peer_is_treated_as_remote(self):
        """No peer info → fail safe toward refusal, not exposure."""
        assert is_local_request(_request(None)) is False


class TestRequireOwner:
    def test_local_caller_passes(self, temp_config):
        # Does not raise.
        require_owner(_request("127.0.0.1"), "pull")

    def test_remote_caller_refused_by_default(self, temp_config):
        with pytest.raises(HTTPException) as exc:
            require_owner(_request("203.0.113.7"), "pull")
        assert exc.value.status_code == 403
        assert exc.value.detail["code"] == "remote_admin_forbidden"
        assert "pull" in exc.value.detail["error"]

    def test_remote_caller_allowed_when_opted_in(self, temp_config):
        temp_config.allow_remote_pull = True
        # Does not raise once the owner enables remote administration.
        require_owner(_request("203.0.113.7"), "pull")

    def test_operation_name_is_surfaced(self, temp_config):
        with pytest.raises(HTTPException) as exc:
            require_owner(_request("10.0.0.9"), "push")
        assert "push" in exc.value.detail["error"]


def _license(risk: LicenseRisk) -> LicenseInfo:
    return LicenseInfo(
        license_id="x",
        license_name="X",
        risk=risk,
        restrictions=[],
        url="https://example.test",
        gated=False,
    )


class TestPolicyAllows:
    def test_permissive_policy_only_allows_permissive(self):
        assert policy_allows(_license(LicenseRisk.PERMISSIVE), "permissive") is True
        for risk in (
            LicenseRisk.CONDITIONAL,
            LicenseRisk.NON_COMMERCIAL,
            LicenseRisk.RESTRICTED,
            LicenseRisk.UNKNOWN,
        ):
            assert policy_allows(_license(risk), "permissive") is False

    def test_conditional_policy_is_cumulative(self):
        assert policy_allows(_license(LicenseRisk.PERMISSIVE), "conditional") is True
        assert policy_allows(_license(LicenseRisk.CONDITIONAL), "conditional") is True
        assert policy_allows(_license(LicenseRisk.NON_COMMERCIAL), "conditional") is False

    def test_all_policy_allows_everything(self):
        for risk in LicenseRisk:
            assert policy_allows(_license(risk), "all") is True

    def test_unknown_policy_string_falls_back_to_permissive(self):
        assert policy_allows(_license(LicenseRisk.PERMISSIVE), "banana") is True
        assert policy_allows(_license(LicenseRisk.CONDITIONAL), "banana") is False

    @pytest.mark.parametrize("policy", ["", "  CONDITIONAL  ", "All"])
    def test_policy_string_is_normalised(self, policy):
        # Empty → permissive; whitespace/case tolerated.
        expected = policy.strip().lower() in ("conditional", "all")
        assert policy_allows(_license(LicenseRisk.CONDITIONAL), policy) is expected
