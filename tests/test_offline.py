# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""HFL with no network: local paths work, Hub paths say "offline".

The independence thesis makes this a first-class state rather than an
error. A local runner that needs the network to list its own models, or
that answers a missing Wi-Fi with ``[Errno 8] nodename nor servname
provided`` and an HTTP 500, is not delivering on the premise.

Measured before these tests existed, with DNS failing:

* every local path already worked — ``hfl list``, ``/healthz``,
  ``/api/tags`` and friends in well under a second, no network touched;
* every Hub path failed fast, but said the wrong thing: raw errnos in the
  CLI, **500** from ``/api/pull`` as if the server had broken, and from
  smart-pull a claim that "the Hub reports no file sizes" about a repo
  nobody had reached — a false statement introduced by the MoE work.

So the tests below pin both halves. The network is removed by failing
DNS for every non-loopback host, which is what a laptop with Wi-Fi off
looks like, and makes each case fail in milliseconds rather than waiting
out a real connect timeout.
"""

from __future__ import annotations

import errno
import socket

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from hfl.hub.connectivity import HubUnreachableError, is_network_error

LOCAL_PEER = ("127.0.0.1", 5555)
_LOOPBACK = {"127.0.0.1", "::1", "localhost", "testserver", None}


@pytest.fixture
def no_network(monkeypatch):
    """Fail name resolution for anything that is not this machine."""
    real = socket.getaddrinfo

    def getaddrinfo(host, *args, **kwargs):
        if host in _LOOPBACK:
            return real(host, *args, **kwargs)
        raise socket.gaierror(8, "nodename nor servname provided, or not known")

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


@pytest.fixture
def client(temp_config, no_network):
    from hfl.api.state import reset_state

    reset_state()
    yield TestClient(app_under_test(), client=LOCAL_PEER)
    reset_state()


def app_under_test():
    from hfl.api.server import app

    return app


runner = CliRunner()


def _cli(*args):
    from hfl.cli.main import app

    return runner.invoke(app, list(args))


class TestLocalPathsNeedNoNetwork:
    """The half that was already true, now asserted so it stays true."""

    @pytest.mark.parametrize("args", [("version",), ("list",), ("sessions", "list")])
    def test_cli(self, temp_config, no_network, args):
        result = _cli(*args)
        assert result.exit_code == 0, result.stdout

    @pytest.mark.parametrize("path", ["/healthz", "/api/tags", "/api/version", "/api/ps"])
    def test_api(self, client, path):
        assert client.get(path).status_code == 200


class TestHubPathsSayOffline:
    """The half that was false: failing fast is not the same as failing clearly."""

    @pytest.mark.parametrize("args", [("search", "llama"), ("pull", "acme/definitely-not-here")])
    def test_cli_names_the_cause_and_the_remedy(self, temp_config, no_network, args):
        result = _cli(*args)
        assert result.exit_code == 1
        assert "huggingface.co" in result.stdout
        assert "hfl list" in result.stdout, (
            "the message must say what still works, not only what broke"
        )
        assert "[Errno" not in result.stdout, (
            "the socket layer's wording reached the user instead of 'offline'"
        )

    def test_pull_is_503_not_500(self, client):
        """Offline is an unavailable upstream. 500 claimed HFL had broken."""
        response = client.post("/api/pull", json={"model": "acme/x", "stream": False})
        assert response.status_code == 503
        body = response.json()
        assert body["code"] == "hub_unreachable"
        assert "offline" in body["error"]

    def test_streamed_pull_carries_the_code(self, client):
        import json

        response = client.post("/api/pull", json={"model": "acme/x", "stream": True})
        events = [json.loads(line) for line in response.text.splitlines() if line]
        assert events[-1]["status"] == "error"
        assert events[-1]["code"] == "hub_unreachable"

    def test_discover_says_offline(self, client):
        response = client.get("/api/discover", params={"q": "llama"})
        assert response.status_code == 503
        assert "huggingface.co" in response.json()["detail"]

    def test_smart_pull_says_offline_not_no_sizes(self, client):
        """The false statement the MoE work introduced.

        A name with no parameter count sends the planner to the Hub for
        file sizes. Offline it used to answer "the Hub reports no file
        sizes" — about a repo it never reached.
        """
        response = client.post(
            "/api/pull/smart", json={"model": "acme/no-size-in-name", "stream": False}
        )
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert "no file sizes" not in detail
        assert "huggingface.co" in detail

    def test_smart_pull_never_blames_the_repos(self, client):
        """A name the size CAN be read from skips the size lookup and goes
        straight to probing candidate repos. Those probes used to swallow
        every exception, so offline each candidate became "not on Hub" and
        the plan concluded nothing fits."""
        response = client.post("/api/pull/smart", json={"model": "acme/Thing-7B", "stream": False})
        detail = response.json()["detail"]
        assert response.status_code == 503
        assert "not on Hub" not in detail
        assert "fits" not in detail


class TestClassification:
    """What counts as "offline" — and, just as much, what does not."""

    @pytest.mark.parametrize(
        "exc",
        [
            socket.gaierror(8, "nodename nor servname provided"),
            socket.timeout("timed out"),
            ConnectionRefusedError(),
            ConnectionResetError(),
            OSError(errno.ENETUNREACH, "Network is unreachable"),
            OSError(errno.EHOSTUNREACH, "No route to host"),
            HubUnreachableError("x"),
        ],
    )
    def test_network_failures_are_recognised(self, exc):
        assert is_network_error(exc)

    @pytest.mark.parametrize(
        "exc",
        [
            FileNotFoundError(errno.ENOENT, "no such file"),
            PermissionError(errno.EACCES, "denied"),
            OSError(errno.ENOSPC, "No space left on device"),
            ValueError("bad input"),
            RuntimeError("anything"),
        ],
    )
    def test_local_failures_are_not_called_offline(self, exc):
        """A missing file is an OSError too. Calling it "offline" would send
        the user chasing their Wi-Fi."""
        assert not is_network_error(exc)

    def test_an_http_error_status_is_an_answer_not_an_outage(self):
        httpx = pytest.importorskip("httpx")
        request = httpx.Request("GET", "https://huggingface.co/api/models/x")
        response = httpx.Response(404, request=request)
        assert not is_network_error(
            httpx.HTTPStatusError("404", request=request, response=response)
        )

    def test_httpx_connect_error_is_offline(self):
        httpx = pytest.importorskip("httpx")
        assert is_network_error(httpx.ConnectError("no route"))

    def test_the_chain_is_followed(self):
        """huggingface_hub and httpx wrap the socket error they caught."""
        try:
            try:
                raise socket.gaierror(8, "nodename")
            except socket.gaierror as inner:
                raise RuntimeError("wrapped by a library") from inner
        except RuntimeError as outer:
            assert is_network_error(outer)

    def test_a_cyclic_chain_terminates(self):
        """The error path must not hang while trying to explain an error."""
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__cause__ = b
        b.__cause__ = a
        assert is_network_error(a) is False


class TestUnreachableIsNotEmpty:
    """ "Could not reach" and "reached, found nothing" call for opposite advice."""

    def test_hub_file_sizes_raise_when_unreachable(self):
        from unittest.mock import MagicMock

        from hfl.hub.params import total_b_from_hub_files

        api = MagicMock()
        api.model_info = MagicMock(side_effect=socket.gaierror(8, "nodename"))
        with pytest.raises(HubUnreachableError):
            total_b_from_hub_files(api, "x/y")

    def test_other_failures_still_return_none(self):
        """Pre-existing contract, kept: a non-network failure is 'could not
        determine', not an outage."""
        from unittest.mock import MagicMock

        from hfl.hub.params import total_b_from_hub_files

        api = MagicMock()
        api.model_info = MagicMock(side_effect=RuntimeError("500 from the Hub"))
        assert total_b_from_hub_files(api, "x/y") is None

    def test_repo_probe_raises_instead_of_answering_not_found(self):
        from unittest.mock import MagicMock

        from hfl.hub.smart_pull import _repo_exists

        api = MagicMock()
        api.model_info = MagicMock(side_effect=ConnectionRefusedError())
        with pytest.raises(HubUnreachableError):
            _repo_exists(api, "x/y")

    def test_a_real_404_is_still_not_found(self):
        from unittest.mock import MagicMock

        from hfl.hub.smart_pull import _repo_exists

        api = MagicMock()
        api.model_info = MagicMock(side_effect=RuntimeError("404 Repository Not Found"))
        assert _repo_exists(api, "x/y") is False


# ----------------------------------------------------------------------
# A network that drops packets: every Hub call must still end
# ----------------------------------------------------------------------

# Run in a fresh interpreter: the fix is armed at ``import hfl`` and acts
# when huggingface_hub loads its HTTP layer, which in this process has
# long happened. The subprocess is also the harness's own kill switch —
# a regression here is a hang, and a hang must not take the suite with it.
_BLACKHOLE = """
import socket, sys, time

def getaddrinfo(host, port, *a, **k):
    if host in ("127.0.0.1", "::1", "localhost"):
        return _real_gai(host, port, *a, **k)
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.1", port))]

def connect(self, addr):
    # Packets vanish: wait out whatever timeout the caller set, as a real
    # blackhole does. None means forever, capped so the child cannot outlive
    # the parent's watchdog by much.
    t = self.gettimeout()
    time.sleep(30 if t is None else t)
    raise socket.timeout("timed out")

_real_gai = socket.getaddrinfo
socket.getaddrinfo = getaddrinfo
socket.socket.connect = connect
"""


def _run_child(body: str, timeout: float = 25):
    import subprocess
    import sys
    import textwrap

    return subprocess.run(
        [sys.executable, "-c", _BLACKHOLE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**__import__("os").environ, "HF_HUB_DISABLE_TELEMETRY": "1"},
    )


_CALL_THE_HUB = """
import time
from huggingface_hub import HfApi
from hfl.hub.connectivity import is_network_error
start = time.monotonic()
try:
    HfApi().model_info("acme/x")
    print("RESULT no-error")
except Exception as exc:
    print("RESULT", is_network_error(exc), round(time.monotonic() - start, 2))
"""


class TestBlackholedHubCallsEnd:
    def test_an_api_call_times_out_instead_of_hanging(self):
        """``model_info`` passes ``timeout=None``: before the fix it waited
        for ever. With the bound it fails in one connect timeout, and as
        "offline"."""
        child = _run_child(
            "import hfl, hfl.hub.timeouts as t\nt.HUB_CONNECT_TIMEOUT = 0.3\n" + _CALL_THE_HUB
        )
        line = [x for x in child.stdout.splitlines() if x.startswith("RESULT")]
        assert line, child.stderr[-2000:]
        _, offline, elapsed = line[0].split()
        assert offline == "True"
        assert float(elapsed) < 5

    def test_the_bound_also_applies_when_huggingface_hub_loaded_first(self):
        """A caller that imported huggingface_hub before hfl gets it too."""
        child = _run_child(
            "import huggingface_hub.utils._http\n"
            "import hfl, hfl.hub.timeouts as t\nt.HUB_CONNECT_TIMEOUT = 0.3\n" + _CALL_THE_HUB
        )
        assert "RESULT True" in child.stdout, child.stderr[-2000:]

    def test_the_control_hangs_without_hfl(self):
        """The same call without ``import hfl`` must NOT finish. If it did,
        the blackhole above would not be simulating anything, and the two
        tests above would pass with no fix at all."""
        import subprocess

        with pytest.raises(subprocess.TimeoutExpired):
            _run_child(_CALL_THE_HUB.replace("from hfl.hub.connectivity", "#"), timeout=4)


class TestBoundDetails:
    def test_explicit_timeouts_are_kept(self):
        from types import SimpleNamespace

        from hfl.hub.timeouts import HUB_CONNECT_TIMEOUT, HUB_READ_TIMEOUT, _fill_unbounded

        request = SimpleNamespace(
            extensions={"timeout": {"connect": None, "read": 3.0, "write": None, "pool": 7.0}}
        )
        _fill_unbounded(request)
        assert request.extensions["timeout"] == {
            "connect": HUB_CONNECT_TIMEOUT,
            "read": 3.0,
            "write": HUB_READ_TIMEOUT,
            "pool": 7.0,
        }

    def test_a_custom_client_factory_is_left_alone(self):
        """Someone who configured a proxy or CA bundle keeps their client."""
        child = _run_child(
            """
            import httpx
            from huggingface_hub import set_client_factory, get_session
            mine = httpx.Client()
            set_client_factory(lambda: mine)
            import hfl, hfl.hub.timeouts as t
            t.install_hub_timeouts()
            print("RESULT", get_session() is mine,
                  t._fill_unbounded in get_session().event_hooks["request"])
            """
        )
        assert "RESULT True False" in child.stdout, child.stderr[-2000:]

    def test_import_hfl_does_not_load_the_network_stack(self):
        """The hook is armed without importing httpx or huggingface_hub:
        ``hfl version`` should not pay ~85 ms for a network it never uses."""
        child = _run_child(
            """
            import sys
            import hfl
            print("RESULT", "httpx" in sys.modules, "huggingface_hub" in sys.modules)
            """
        )
        assert "RESULT False False" in child.stdout, child.stderr[-2000:]
