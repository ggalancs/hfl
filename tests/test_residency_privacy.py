# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""What a remote client learns when a load is refused or has to wait.

Found in the pre-release audit of the memory-budget work: the 503 for a
busy wait named the models OTHER clients were using at that moment, and
the 507 and /api/ps gave any client the host's RAM figures. The owner (a
loopback peer) still gets everything; a remote peer gets the outcome.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hfl.api.exception_handlers import register_exception_handlers
from hfl.engine.residency import AdmissionPlan
from hfl.exceptions import MemoryBudgetExceededError, ModelsBusyError

GB = 10**9


def _app():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/refused")
    def refused():
        plan = AdmissionPlan(
            False,
            reason="too_big",
            limit=108 * GB,
            used_now=60 * GB,
            used_after=150 * GB,
            floor=150 * GB,
        )
        raise MemoryBudgetExceededError(
            "big-model", needed=90 * GB, plan=plan, total=128 * GB, budget=0.85
        )

    @app.get("/busy")
    def busy():
        raise ModelsBusyError("new-model", ["alice-private-finetune", "bob-model"])

    return app


@pytest.fixture
def remote():
    return TestClient(_app())  # client.host == "testclient": a remote peer


@pytest.fixture
def owner():
    return TestClient(_app(), client=("127.0.0.1", 50000))


def test_a_busy_wait_does_not_name_other_clients_models(remote):
    response = remote.get("/busy")
    assert response.status_code == 503
    text = response.text
    assert "alice-private-finetune" not in text and "bob-model" not in text
    assert "new-model" in text  # the client's own model is fine to echo


def test_a_refusal_does_not_give_out_host_memory(remote):
    response = remote.get("/refused")
    assert response.status_code == 507
    assert "GB" not in response.text and "HFL_MEMORY_BUDGET" not in response.text
    assert "big-model" in response.text


def test_the_owner_still_gets_the_numbers_and_the_names(owner):
    assert "alice-private-finetune" in owner.get("/busy").text
    refused = owner.get("/refused").text
    assert "GB" in refused and "HFL_MEMORY_BUDGET" in refused


def test_these_are_not_logged_as_unhandled_errors(remote, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="hfl.api.exception_handlers"):
        remote.get("/busy")
    assert "Unhandled" not in caplog.text
    assert "alice-private-finetune" in caplog.text  # the log keeps the detail
