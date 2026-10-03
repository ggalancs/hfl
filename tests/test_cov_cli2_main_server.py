# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""CLI commands that talk to a running HFL server, or load a model in
process: ``ps``, ``create``, ``lora``, ``snapshot``, ``mcp``, ``verify``,
``bench``, ``pull-smart``.

The server is never real: ``httpx`` is replaced by stubs that answer as
the server does (or fail the way a network does), and the model loader by
an async fake. What is checked is what the user sees and the exit code.
"""

from __future__ import annotations

import json
import sys
import types
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from typer.testing import CliRunner

from hfl.cli import main
from hfl.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _fixed_port(monkeypatch):
    from hfl.config import config

    monkeypatch.setattr(config, "port", 18555)
    monkeypatch.delenv("HFL_API_KEY", raising=False)


class _Resp:
    def __init__(self, status_code=200, payload=None, text="", error=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self._error = error

    def json(self):
        if self._error:
            raise self._error
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://x")
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=request,
                response=httpx.Response(self.status_code, request=request),
            )


def _raise(exc):
    def fn(*a, **k):
        raise exc

    return fn


# ----------------------------------------------------------------------
# ps
# ----------------------------------------------------------------------


class TestPs:
    def test_unreachable_server_says_how_to_start_one(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", _raise(httpx.ConnectError("refused")))
        result = runner.invoke(app, ["ps"])
        assert result.exit_code == 1
        assert "Cannot reach HFL server at http://127.0.0.1:18555/api/ps" in result.stdout
        assert "hfl serve" in result.stdout

    def test_a_server_error_is_exit_1(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(500))
        result = runner.invoke(app, ["ps"])
        assert result.exit_code == 1
        assert "Server error:" in result.stdout

    def test_rows_classify_processor_and_size(self, monkeypatch):
        gib = 1024**3
        payload = {
            "models": [
                {"name": "all-gpu", "size": 4 * gib, "size_vram": 4 * gib, "digest": "a" * 64},
                {"name": "on-cpu", "size": 2 * gib, "size_vram": 0, "expires_at": "soon"},
                {"name": "split", "size": 8 * gib, "size_vram": 2 * gib},
                {"name": "tiny", "size": 5 * 1024**2},
            ],
            "memory": {
                "total_bytes": 64 * gib,
                "in_use_bytes": 16 * gib,
                "in_use_percent": 25,
                "budget_percent": 80,
                "budget_bytes": 51 * gib,
                "free_within_budget_bytes": 35 * gib,
                "models_bytes": 14 * gib,
                "gpu": {
                    "total_bytes": 24 * gib,
                    "in_use_bytes": 6 * gib,
                    "in_use_percent": 25,
                    "free_within_budget_bytes": 18 * gib,
                },
            },
        }
        monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(200, payload))
        result = runner.invoke(app, ["ps"], terminal_width=200)
        assert result.exit_code == 0, result.stdout
        rows = {line.split()[1]: line for line in result.stdout.splitlines() if "│" in line[:2]}
        assert "GPU" in rows["all-gpu"] and "4.0 GB" in rows["all-gpu"]
        assert "aaaaaaaaaaaa" in rows["all-gpu"]
        assert "CPU" in rows["on-cpu"] and "soon" in rows["on-cpu"]
        assert "GPU" in rows["split"]
        assert "5 MB" in rows["tiny"] and "CPU" in rows["tiny"]
        assert "Memory: 16.0 of 64.0 GB in use (25%)" in result.stdout
        assert "GPU: 6.0 of 24.0 GB in use (25%) · room left: 18.0 GB" in result.stdout

    def test_no_models_and_an_old_server_without_memory(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(200, {"models": []}))
        result = runner.invoke(app, ["ps"])
        assert result.exit_code == 0
        assert "No models loaded" in result.stdout
        assert "Memory:" not in result.stdout


# ----------------------------------------------------------------------
# create
# ----------------------------------------------------------------------


class _Stream:
    def __init__(self, lines=(), status=200, error=None):
        self.lines = list(lines)
        self.status = status
        self.error = error
        self.sent = None

    @contextmanager
    def __call__(self, method, url, json, timeout):
        if self.error:
            raise self.error
        self.sent = (method, url, json)
        response = _Resp(self.status)
        response.iter_lines = lambda: iter(self.lines)
        yield response


@pytest.fixture
def modelfile(tmp_path):
    path = tmp_path / "Modelfile"
    path.write_text("FROM base\nSYSTEM be brief\n")
    return path


class TestCreate:
    def _run(self, monkeypatch, stream, modelfile):
        monkeypatch.setattr(httpx, "stream", stream)
        return runner.invoke(app, ["create", "mine", "-f", str(modelfile)])

    def test_success_streams_each_status(self, monkeypatch, modelfile):
        stream = _Stream(
            [
                json.dumps({"status": "reading modelfile"}),
                "",
                "not json at all",
                json.dumps({"status": "success"}),
            ]
        )
        result = self._run(monkeypatch, stream, modelfile)
        assert result.exit_code == 0, result.stdout
        assert "reading modelfile" in result.stdout
        assert "not json at all" in result.stdout
        method, url, payload = stream.sent
        assert (method, url) == ("POST", "http://127.0.0.1:18555/api/create")
        assert payload == {
            "model": "mine",
            "modelfile": "FROM base\nSYSTEM be brief\n",
            "stream": True,
        }

    def test_an_error_event_is_exit_1(self, monkeypatch, modelfile):
        stream = _Stream([json.dumps({"error": "base not found"})])
        result = self._run(monkeypatch, stream, modelfile)
        assert result.exit_code == 1
        assert "Error: base not found" in result.stdout

    def test_a_stream_that_ends_before_success_is_a_failure(self, monkeypatch, modelfile):
        stream = _Stream([json.dumps({"status": "creating"})])
        result = self._run(monkeypatch, stream, modelfile)
        assert result.exit_code == 1
        assert "nothing was saved" in result.stdout

    def test_unreachable_server(self, monkeypatch, modelfile):
        result = self._run(monkeypatch, _Stream(error=httpx.ConnectError("no")), modelfile)
        assert result.exit_code == 1
        assert "Cannot reach HFL server" in result.stdout

    def test_http_error_status(self, monkeypatch, modelfile):
        result = self._run(monkeypatch, _Stream(status=403), modelfile)
        assert result.exit_code == 1
        assert "Server error:" in result.stdout


# ----------------------------------------------------------------------
# _server_request, lora, snapshot
# ----------------------------------------------------------------------


class _Requests:
    """httpx.request stand-in that records calls and answers in order."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[tuple] = []

    def __call__(self, method, url, json=None, headers=None, timeout=None):
        self.calls.append((method, url, json, headers))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class TestServerRequest:
    def test_sends_the_api_key(self, monkeypatch):
        requests = _Requests(_Resp(200, {"ok": True}))
        monkeypatch.setattr(httpx, "request", requests)
        monkeypatch.setenv("HFL_API_KEY", "s3cret")
        assert main._server_request("GET", "h", 1234, "/api/x") == {"ok": True}
        assert requests.calls[0][1] == "http://h:1234/api/x"
        assert requests.calls[0][3] == {"Authorization": "Bearer s3cret"}

    def test_unreachable(self, monkeypatch, capsys):
        monkeypatch.setattr(httpx, "request", _Requests(httpx.ConnectError("no")))
        with pytest.raises(main.typer.Exit) as caught:
            main._server_request("GET", "h", None, "/api/x")
        assert caught.value.exit_code == 1
        assert "Cannot reach HFL server at http://h:18555/api/x" in capsys.readouterr().out

    def test_other_transport_errors(self, monkeypatch, capsys):
        monkeypatch.setattr(httpx, "request", _Requests(httpx.ReadTimeout("slow [x]")))
        with pytest.raises(main.typer.Exit):
            main._server_request("GET", "h", 1, "/p")
        assert "Server error: slow [x]" in capsys.readouterr().out

    def test_error_detail_or_error_or_text(self, monkeypatch, capsys):
        monkeypatch.setattr(
            httpx,
            "request",
            _Requests(
                _Resp(404, {"detail": "no such model"}),
                _Resp(409, {"error": "busy"}),
                _Resp(502, error=ValueError("not json"), text="bad gateway"),
            ),
        )
        for _ in range(3):
            with pytest.raises(main.typer.Exit):
                main._server_request("POST", "h", 1, "/p", {"a": 1})
        out = capsys.readouterr().out
        assert "404: no such model" in out
        assert "409: busy" in out
        assert "502: bad gateway" in out


class TestLora:
    @pytest.mark.parametrize(
        ("args", "said"),
        [
            (["lora", "frobnicate"], "Action must be one of"),
            (["lora", "apply"], "Model name is required"),
            (["lora", "apply", "m"], "--path is required"),
            (["lora", "remove", "m"], "--id is required"),
        ],
    )
    def test_usage_errors(self, monkeypatch, args, said):
        requests = _Requests()
        monkeypatch.setattr(httpx, "request", requests)
        result = runner.invoke(app, args)
        assert result.exit_code == 1
        assert said in result.stdout
        assert requests.calls == []

    def test_apply(self, monkeypatch):
        requests = _Requests(_Resp(200, {"adapter_id": "ad-1"}))
        monkeypatch.setattr(httpx, "request", requests)
        result = runner.invoke(
            app, ["lora", "apply", "m", "--path", "a.safetensors", "--scale", "0.5"]
        )
        assert result.exit_code == 0
        assert "Applied adapter id=ad-1" in result.stdout
        method, url, body, _ = requests.calls[0]
        assert (method, url) == ("POST", "http://127.0.0.1:18555/api/lora/apply")
        assert body == {"model": "m", "lora_path": "a.safetensors", "scale": 0.5, "name": None}

    def test_remove(self, monkeypatch):
        requests = _Requests(_Resp(200, {}))
        monkeypatch.setattr(httpx, "request", requests)
        result = runner.invoke(app, ["lora", "remove", "m", "--id", "ad-1"])
        assert result.exit_code == 0 and "Removed" in result.stdout
        assert requests.calls[0][2] == {"model": "m", "adapter_id": "ad-1"}

    def test_list_all_and_one_model(self, monkeypatch):
        adapter = {"adapter_id": "ad-1", "name": None, "path": "/a", "scale": 0.7}
        requests = _Requests(_Resp(200, {"adapters": []}), _Resp(200, {"adapters": [adapter]}))
        monkeypatch.setattr(httpx, "request", requests)
        empty = runner.invoke(app, ["lora", "list"])
        assert "No adapters active." in empty.stdout
        listed = runner.invoke(app, ["lora", "list", "m"], terminal_width=150)
        assert listed.exit_code == 0
        assert "ad-1" in listed.stdout and "0.70" in listed.stdout
        assert [c[1].rsplit(":18555", 1)[1] for c in requests.calls] == ["/api/lora", "/api/lora/m"]


class TestSnapshot:
    @pytest.mark.parametrize(
        ("args", "said"),
        [
            (["snapshot", "freeze"], "Action must be one of"),
            (["snapshot", "save"], "Model is required"),
            (["snapshot", "delete"], "--name is required"),
        ],
    )
    def test_usage_errors(self, args, said):
        result = runner.invoke(app, args)
        assert result.exit_code == 1 and said in result.stdout

    def test_list(self, monkeypatch):
        entry = {"name": "warm", "model": "m", "tokens": 42, "bytes": 123456}
        monkeypatch.setattr(
            httpx,
            "request",
            _Requests(_Resp(200, {"snapshots": []}), _Resp(200, {"snapshots": [entry]})),
        )
        assert "No snapshots saved." in runner.invoke(app, ["snapshot", "list"]).stdout
        listed = runner.invoke(app, ["snapshot", "list"], terminal_width=150)
        assert "warm" in listed.stdout and "123,456" in listed.stdout

    def test_delete_save_load(self, monkeypatch):
        requests = _Requests(
            _Resp(200, {}),
            _Resp(200, {"tokens": 10, "bytes": 2048}),
            _Resp(200, {"tokens": 10}),
        )
        monkeypatch.setattr(httpx, "request", requests)
        deleted = runner.invoke(app, ["snapshot", "delete", "--name", "w"])
        assert "Deleted snapshot 'w'" in deleted.stdout
        saved = runner.invoke(app, ["snapshot", "save", "m", "--name", "w"])
        assert "Saved 'w' — tokens=10 bytes=2,048" in saved.stdout
        loaded = runner.invoke(app, ["snapshot", "load", "m", "--name", "w"])
        assert "Restored 'w' — tokens=10" in loaded.stdout
        assert [(c[0], c[1].rsplit(":18555", 1)[1]) for c in requests.calls] == [
            ("DELETE", "/api/snapshot/w"),
            ("POST", "/api/snapshot/save"),
            ("POST", "/api/snapshot/load"),
        ]
        assert requests.calls[1][2] == {"model": "m", "name": "w"}


# ----------------------------------------------------------------------
# mcp
# ----------------------------------------------------------------------


class TestMcp:
    @pytest.fixture
    def client(self, monkeypatch):
        import hfl.mcp.client as mcp_client

        fake = MagicMock()
        fake.connect = AsyncMock()
        fake.disconnect = AsyncMock()
        monkeypatch.setattr(mcp_client, "get_client", lambda: fake)
        return fake

    @pytest.fixture
    def server(self, monkeypatch):
        import hfl.mcp.server as mcp_server

        stdio, sse = AsyncMock(), AsyncMock()
        monkeypatch.setattr(mcp_server, "serve_stdio", stdio)
        monkeypatch.setattr(mcp_server, "serve_sse", sse)
        return types.SimpleNamespace(stdio=stdio, sse=sse, module=mcp_server)

    def _tool(self, name):
        return types.SimpleNamespace(qualified_name=name, description=f"{name} does things")

    def test_list(self, client):
        client.list_tools.return_value = []
        assert "No MCP servers connected." in runner.invoke(app, ["mcp", "list"]).stdout
        client.list_tools.return_value = [self._tool("fs.read")]
        result = runner.invoke(app, ["mcp", "list"])
        assert result.exit_code == 0 and "fs.read" in result.stdout

    def test_connect(self, client):
        client.connect.return_value = [self._tool("fs.read"), self._tool("fs.write")]
        result = runner.invoke(app, ["mcp", "connect", "fs", "stdio://npx server"])
        assert result.exit_code == 0
        assert "Connected fs (2 tools)" in result.stdout
        client.connect.assert_awaited_once_with("fs", "stdio://npx server")

    def test_connect_and_disconnect_need_their_arguments(self, client):
        assert runner.invoke(app, ["mcp", "connect", "fs"]).exit_code == 1
        assert runner.invoke(app, ["mcp", "disconnect"]).exit_code == 1
        client.connect.assert_not_awaited()
        client.disconnect.assert_not_awaited()

    def test_disconnect(self, client):
        result = runner.invoke(app, ["mcp", "disconnect", "fs"])
        assert result.exit_code == 0 and "Disconnected fs" in result.stdout
        client.disconnect.assert_awaited_once_with("fs")

    def test_unknown_action(self, client):
        result = runner.invoke(app, ["mcp", "dance"])
        assert result.exit_code == 1 and "Unknown action: dance" in result.stdout

    def test_serve_stdio_with_capabilities(self, client, server):
        result = runner.invoke(app, ["mcp", "serve", "--capabilities", "web_search, ,web_fetch"])
        assert result.exit_code == 0
        server.stdio.assert_awaited_once_with(["web_search", "web_fetch"])

    def test_serve_sse(self, client, server):
        result = runner.invoke(app, ["mcp", "serve", "--transport", "sse", "--port", "9999"])
        assert result.exit_code == 0
        server.sse.assert_awaited_once_with("127.0.0.1", 9999, None)

    def test_serve_unknown_transport(self, client, server):
        result = runner.invoke(app, ["mcp", "serve", "--transport", "carrier-pigeon"])
        assert result.exit_code == 1 and "Unknown transport" in result.stdout

    def test_serve_without_the_mcp_package(self, client, server):
        server.stdio.side_effect = server.module.MCPServerUnavailableError("SDK missing")
        result = runner.invoke(app, ["mcp", "serve"])
        assert result.exit_code == 1
        assert "MCP unavailable: SDK missing" in result.stdout

    def test_client_unavailable_and_connection_errors(self, client):
        import hfl.mcp.client as mcp_client

        client.connect.side_effect = mcp_client.MCPClientUnavailableError("no mcp")
        result = runner.invoke(app, ["mcp", "connect", "fs", "stdio://x"])
        assert result.exit_code == 1 and "MCP unavailable: no mcp" in result.stdout
        client.connect.side_effect = mcp_client.MCPConnectionError("fs", "refused")
        result = runner.invoke(app, ["mcp", "connect", "fs", "stdio://x"])
        assert result.exit_code == 1 and "MCP error: MCP server 'fs': refused" in result.stdout

    # The SDK's real message names the extra to install: `pip install
    # 'hfl[mcp]'`. Printed through Rich markup unescaped, "[mcp]" is read as
    # a style tag and dropped, so the user is told to run `pip install 'hfl'`,
    # which does not install the SDK.
    _SDK_MISSING = "The MCP SDK is not installed. `pip install 'hfl[mcp]'` adds it."

    def test_serve_keeps_the_install_hint(self, client, server):
        server.stdio.side_effect = server.module.MCPServerUnavailableError(self._SDK_MISSING)
        result = runner.invoke(app, ["mcp", "serve"], terminal_width=200)
        assert "pip install 'hfl[mcp]'" in result.stdout

    def test_connect_keeps_the_install_hint(self, client):
        import hfl.mcp.client as mcp_client

        client.connect.side_effect = mcp_client.MCPClientUnavailableError(self._SDK_MISSING)
        result = runner.invoke(app, ["mcp", "connect", "fs", "stdio://x"], terminal_width=200)
        assert "pip install 'hfl[mcp]'" in result.stdout


# ----------------------------------------------------------------------
# verify / bench (an in-process model, faked)
# ----------------------------------------------------------------------


def _loader(monkeypatch, engine=None, error=None):
    import hfl.api.model_loader as loader

    async def load_llm(name):
        if error:
            raise error
        return engine, types.SimpleNamespace(name=name)

    monkeypatch.setattr(loader, "load_llm", load_llm)


class TestVerify:
    def test_missing_model(self, monkeypatch):
        from hfl.exceptions import ModelNotFoundError

        _loader(monkeypatch, error=ModelNotFoundError("ghost"))
        result = runner.invoke(app, ["verify", "ghost"])
        assert result.exit_code == 1
        assert "Model not found: ghost" in result.stdout
        assert "hfl list" in result.stdout

    def test_no_engine(self, monkeypatch):
        _loader(monkeypatch, engine=None)
        result = runner.invoke(app, ["verify", "m"])
        assert result.exit_code == 1 and "Engine not available" in result.stdout

    def _verdict(self, monkeypatch, passed):
        import hfl.engine.verifier as verifier

        checks = [
            types.SimpleNamespace(name="tokenizer", passed=True, detail="ok", skipped=False),
            types.SimpleNamespace(name="embedding", passed=False, detail="n/a", skipped=True),
            types.SimpleNamespace(name="smoke", passed=passed, detail="said hi", skipped=False),
        ]
        result = types.SimpleNamespace(
            model="m", duration_ms=12.34, overall_pass=passed, checks=checks
        )
        monkeypatch.setattr(verifier, "verify_model", lambda engine, manifest: result)

    def test_pass(self, monkeypatch):
        _loader(monkeypatch, engine=object())
        self._verdict(monkeypatch, True)
        result = runner.invoke(app, ["verify", "m"], terminal_width=150)
        assert result.exit_code == 0
        assert "VERIFY PASS m (12.3 ms)" in result.stdout
        assert "SKIP" in result.stdout and "PASS" in result.stdout

    def test_fail_is_exit_1(self, monkeypatch):
        _loader(monkeypatch, engine=object())
        self._verdict(monkeypatch, False)
        result = runner.invoke(app, ["verify", "m"], terminal_width=150)
        assert result.exit_code == 1
        assert "VERIFY FAIL" in result.stdout and "FAIL" in result.stdout


class TestBench:
    def test_bad_lengths(self):
        result = runner.invoke(app, ["bench", "m", "--lengths", "16,abc"])
        assert result.exit_code == 1 and "Invalid --lengths" in result.stdout

    def test_missing_model(self, monkeypatch):
        _loader(monkeypatch, error=FileNotFoundError("gone"))
        result = runner.invoke(app, ["bench", "m"])
        assert result.exit_code == 1 and "Model not found: m" in result.stdout

    def test_no_engine(self, monkeypatch):
        _loader(monkeypatch, engine=None)
        result = runner.invoke(app, ["bench", "m"])
        assert result.exit_code == 1 and "Engine not available" in result.stdout

    def test_streams_runs_and_summarises(self, monkeypatch):
        import hfl.engine.benchmark as benchmark

        _loader(monkeypatch, engine=object())
        seen = {}

        async def stream(engine, model_name, runs_per_length, max_tokens, prompt_lengths):
            seen.update(runs=runs_per_length, tokens=max_tokens, lengths=prompt_lengths)
            yield {
                "status": "starting",
                "model": model_name,
                "runs_per_length": runs_per_length,
                "prompt_lengths": list(prompt_lengths),
            }
            yield {
                "status": "run",
                "prompt_length": 16,
                "ttft_ms": 5.0,
                "total_ms": 50.0,
                "tokens_per_second": 33.333,
            }
            yield {"status": "progress"}
            base = {"status": "summary", "runs": 2, "tps_mean": 30.0, "tps_min": 25.0}
            base["tps_max"] = 35.0
            yield {**base, "prompt_length": 16, "ttft_p50_ms": 5.0, "ttft_p95_ms": 7.25}
            yield {**base, "prompt_length": 32, "ttft_p50_ms": None, "ttft_p95_ms": None}
            yield {"status": "done"}

        monkeypatch.setattr(benchmark, "run_benchmark_stream", stream)
        result = runner.invoke(
            app, ["bench", "m", "--runs", "2", "-t", "8", "--lengths", "16, 32"], terminal_width=160
        )
        assert result.exit_code == 0, result.stdout
        assert seen == {"runs": 2, "tokens": 8, "lengths": (16, 32)}
        assert "Bench m: 2 runs" in result.stdout
        assert "ttft=5.0ms total=50.0ms tps=33.33" in result.stdout
        assert "7.2" in result.stdout and "—" in result.stdout
        assert "30.00" in result.stdout


# ----------------------------------------------------------------------
# pull-smart
# ----------------------------------------------------------------------


class TestPullSmart:
    def _plan(self, monkeypatch, plan=None, error=None):
        import hfl.hub.smart_pull as smart_pull

        def build(model, max_vram_gb):
            if error:
                raise error
            return plan

        monkeypatch.setattr(smart_pull, "build_smart_plan", build)

    @pytest.mark.parametrize(
        ("error", "said"),
        [(ValueError("not a repo"), "not a repo"), (RuntimeError("down"), "Hub unavailable: down")],
    )
    def test_plan_errors(self, monkeypatch, error, said):
        self._plan(monkeypatch, error=error)
        result = runner.invoke(app, ["pull-smart", "org/m"])
        assert result.exit_code == 1 and said in result.stdout

    def _ok_plan(self):
        return types.SimpleNamespace(
            target_repo_id="org/m-GGUF",
            quantization="q4_k_m",
            estimated_vram_gb=4.25,
            reason="fits",
            fallback_chain=["org/m-mlx (not Apple)"],
        )

    def test_pulls_the_chosen_variant(self, monkeypatch):
        import hfl.api.routes_pull as routes_pull

        self._plan(monkeypatch, plan=self._ok_plan())
        pulled = {}

        async def events(repo, quantization):
            pulled["args"] = (repo, quantization)
            yield '{"status": "success"}\n'

        monkeypatch.setattr(routes_pull, "iter_pull_events", events)
        result = runner.invoke(app, ["pull-smart", "org/m"], terminal_width=150)
        assert result.exit_code == 0, result.stdout
        assert pulled["args"] == ("org/m-GGUF", "q4_k_m")
        assert "4.2 GB" in result.stdout or "4.3 GB" in result.stdout
        assert "Skipped candidates" in result.stdout
        assert '{"status": "success"}' in result.stdout

    def test_no_skipped_candidates_says_none(self, monkeypatch):
        import hfl.api.routes_pull as routes_pull

        plan = self._ok_plan()
        plan.fallback_chain = []
        self._plan(monkeypatch, plan=plan)

        async def events(repo, quantization):
            yield "done"

        monkeypatch.setattr(routes_pull, "iter_pull_events", events)
        result = runner.invoke(app, ["pull-smart", "org/m"])
        assert result.exit_code == 0
        assert "Skipped candidates" not in result.stdout and "done" in result.stdout

    def test_a_failed_pull_is_exit_1(self, monkeypatch):
        import hfl.api.routes_pull as routes_pull

        self._plan(monkeypatch, plan=self._ok_plan())

        async def events(repo, quantization):
            raise RuntimeError("disk full")
            yield  # unreachable: makes this an async generator

        monkeypatch.setattr(routes_pull, "iter_pull_events", events)
        result = runner.invoke(app, ["pull-smart", "org/m"])
        assert result.exit_code == 1 and "Pull failed: disk full" in result.stdout


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
def test_module_entry_point_runs_the_cli(monkeypatch):
    """``python -m hfl.cli.main --version`` runs the same CLI."""
    import runpy

    monkeypatch.setattr(sys, "argv", ["hfl", "--version"])
    import hfl.utils.self_exec as self_exec

    monkeypatch.setattr(self_exec, "watch_onefile_launcher", lambda: None)
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("hfl.cli.main", run_name="__main__")
    assert caught.value.code in (0, None)
