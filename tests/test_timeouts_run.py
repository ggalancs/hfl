# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Request timeouts: proof they run, and where from.

Written after a dormant-module scan found that nothing imports
``hfl/api/timeout.py`` — the module ``CLAUDE.md`` names as the project's
timeout mechanism. Two readings fit that fact, and they are opposites: a
superseded duplicate, or a documented feature that does not run. Reading
the code was not enough to choose, so it was measured.

The answer is the first. ``hfl.api.helpers.run_dispatched`` enforces
``config.generation_timeout`` with ``asyncio.wait_for`` and is what the
routers import; ``api/timeout.py`` is an earlier implementation that even
exports a ``run_with_timeout`` under the same name as the live one.
Measured in isolation, the deadline tracks the knob exactly::

    limit 0.25 s -> cut at 0.25 s, HTTP 504
    limit 1.00 s -> cut at 1.00 s, HTTP 504
    limit 2.00 s -> cut at 2.00 s, HTTP 504

A first attempt measured 3.76 s against a 1.0 s limit and looked like a
bug. It was not: the previous probe's ``time.sleep`` thread was still
alive, and ``run_dispatched`` holds the dispatcher slot until a worker
truly exits — deliberately, because a thread inside the engine cannot be
cancelled and a second inference on the shared model would corrupt it.
The queue wait is not part of the generation deadline. Each case below
therefore resets the container and runs alone.

These tests exist because the timeout was enforced by code nobody had
asserted, while the module everyone would look in was inert. That is the
shape of a feature that quietly stops working.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi import HTTPException

import hfl.config as hfl_config
from hfl.core.container import reset_container


@pytest.fixture
def isolated_dispatcher():
    """One probe per event loop: a leftover worker holds the slot."""
    reset_container()
    original = hfl_config.config.generation_timeout
    yield
    hfl_config.config.generation_timeout = original
    reset_container()


@pytest.fixture
def blocking_worker():
    """A worker that outlives its deadline but not the test.

    A plain ``time.sleep(30)`` works and costs 30 seconds: the thread
    cannot be cancelled, so the process waits for it long after the 504
    was raised. Blocking on an Event the test releases keeps the same
    property — the worker is still running when the deadline fires —
    without leaving a zombie behind.
    """
    release = threading.Event()
    entered = threading.Event()

    def worker():
        entered.set()
        release.wait(timeout=30.0)
        return "late"

    yield worker, release, entered
    release.set()


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.parametrize("limit", [0.25, 0.75])
async def test_the_deadline_is_the_configured_one(limit, isolated_dispatcher, blocking_worker):
    """Not merely "a timeout happens" — the value is honoured.

    ``HFL_GENERATION_TIMEOUT`` feeds this. A knob that is read but not
    applied would pass a test that only checked for 504.
    """
    from hfl.api.helpers import run_dispatched

    hfl_config.config.generation_timeout = limit

    worker, release, entered = blocking_worker
    started = time.perf_counter()
    with pytest.raises(HTTPException) as caught:
        await run_dispatched(worker, operation="probe")
    elapsed = time.perf_counter() - started
    release.set()
    assert entered.is_set(), "the worker never started, so nothing was timed out"

    assert caught.value.status_code == 504
    assert elapsed == pytest.approx(limit, abs=0.35), (
        f"deadline of {limit}s fired at {elapsed:.2f}s — the configured value "
        "is not the one being applied"
    )


@pytest.mark.asyncio
@pytest.mark.slow
async def test_the_timeout_reports_a_machine_readable_code(isolated_dispatcher, blocking_worker):
    from hfl.api.helpers import run_dispatched

    worker, release, _entered = blocking_worker
    hfl_config.config.generation_timeout = 0.25
    with pytest.raises(HTTPException) as caught:
        await run_dispatched(worker, operation="inference")
    release.set()

    detail = caught.value.detail
    assert isinstance(detail, dict)
    assert detail.get("code") == "TIMEOUT", "clients branch on the code, not the prose"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_work_inside_the_deadline_is_not_cut(isolated_dispatcher):
    """The other direction, so the guard cannot pass by timing out always."""
    from hfl.api.helpers import run_dispatched

    hfl_config.config.generation_timeout = 2.0
    result = await run_dispatched(lambda: (time.sleep(0.1), "done")[1], operation="quick")
    assert result == "done"


def test_the_live_enforcer_is_helpers_not_the_timeout_module():
    """Names where the mechanism actually lives.

    ``api/timeout.py`` exports a ``run_with_timeout`` with the same name
    as the live one in ``helpers.py``. Anyone reading ``CLAUDE.md`` is
    sent to the inert one. If the enforcement ever moves, this fails and
    the documentation gets corrected with it.
    """
    import inspect

    from hfl.api import helpers

    source = inspect.getsource(helpers.run_dispatched)
    assert "generation_timeout" in source
    assert "wait_for" in source

    routers_importing = 0
    from pathlib import Path

    api_dir = Path(helpers.__file__).parent
    for route in api_dir.glob("routes_*.py"):
        text = route.read_text(encoding="utf-8")
        if "run_dispatched" in text or "run_with_timeout" in text:
            routers_importing += 1
    assert routers_importing >= 5, (
        f"only {routers_importing} routers reach the timeout helpers — "
        "inference paths may have been moved off them"
    )


def test_the_dormant_module_is_not_silently_revived():
    """If someone wires ``api/timeout.py`` up, two enforcers exist at once.

    Two timeout implementations with one config knob is how a deadline
    ends up applied twice, or not at all. Whoever revives it must delete
    the other.
    """
    import ast
    from pathlib import Path

    api_dir = Path(__file__).resolve().parents[1] / "src" / "hfl" / "api"
    importers = []
    for path in (api_dir.parent).rglob("*.py"):
        if path.name == "timeout.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("api.timeout"):
                importers.append(path.name)

    assert not importers, (
        f"{importers} now import api/timeout.py while hfl.api.helpers still "
        "enforces the same config knob. Pick one enforcer and delete the other."
    )
