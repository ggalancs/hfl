# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The one line that says a faster backend exists for this machine.

The selector routes any GGUF to llama.cpp, because MLX cannot ingest
GGUF. That is correct and stays correct — but on Apple Silicon an MLX
*build* of the same model usually exists and is markedly faster,
especially at prompt processing, so serving the slower path in silence
costs the user something they never learn about.

The advisory is deliberately **offline**. Confirming that
``mlx-community/<model>`` exists would put a Hub round-trip in front of
every model load, and loading a local model must not need the network —
that is the whole premise of the project. So it says a build *may* exist
and names the command that checks. These tests hold it to that: the
wording must not claim knowledge it does not have, and the code must not
reach for the network to get it.
"""

from __future__ import annotations

import logging

import pytest

from hfl.engine import selector


@pytest.fixture(autouse=True)
def _forget_previous_advice():
    selector._MLX_ADVISED.clear()
    yield
    selector._MLX_ADVISED.clear()


@pytest.fixture
def apple_silicon_with_mlx(monkeypatch):
    monkeypatch.setattr(selector.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(selector.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(selector, "_mlx_preferred", lambda: True)
    monkeypatch.delenv("HFL_NO_MLX_HINT", raising=False)


def _advise(caplog, path="/m/Qwen2.5-7B-Instruct-Q4_K_M.gguf"):
    with caplog.at_level(logging.INFO, logger="hfl.engine.selector"):
        selector._advise_mlx_alternative(path)
    return caplog.text


class TestWhenItSpeaks:
    def test_names_the_command_that_checks(self, apple_silicon_with_mlx, caplog):
        text = _advise(caplog)
        assert "hfl search mlx-community/Qwen2.5-7B-Instruct" in text, (
            "the advice must be runnable, not a suggestion to go looking"
        )

    def test_strips_the_quantisation_from_the_base_name(self, apple_silicon_with_mlx, caplog):
        """An MLX fork is not named after a GGUF quantisation."""
        text = _advise(caplog, "/m/Llama-3.1-8B-Instruct-Q5_K_M.gguf")
        assert "mlx-community/Llama-3.1-8B-Instruct" in text
        assert "Q5_K_M" not in text.split("mlx-community/")[1]

    def test_says_it_only_once_per_model(self, apple_silicon_with_mlx, caplog):
        path = "/m/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        _advise(caplog, path)
        _advise(caplog, path)
        assert caplog.text.count("mlx-community/") == 1, (
            "a server reloading a model must not repeat this on every load"
        )

    def test_a_different_quantisation_is_a_different_decision(self, apple_silicon_with_mlx, caplog):
        _advise(caplog, "/m/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
        _advise(caplog, "/m/Qwen2.5-7B-Instruct-Q8_0.gguf")
        assert caplog.text.count("mlx-community/") == 2


class TestWhenItStaysQuiet:
    def test_not_on_a_machine_that_cannot_act_on_it(self, monkeypatch, caplog):
        monkeypatch.setattr(selector.platform, "system", lambda: "Linux")
        monkeypatch.setattr(selector, "_mlx_preferred", lambda: True)
        assert "mlx-community/" not in _advise(caplog)

    def test_not_on_intel_macs(self, monkeypatch, caplog):
        monkeypatch.setattr(selector.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(selector.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(selector, "_mlx_preferred", lambda: True)
        assert "mlx-community/" not in _advise(caplog)

    def test_not_when_mlx_is_not_installed(self, monkeypatch, caplog):
        """Advice the user cannot follow today is noise."""
        monkeypatch.setattr(selector.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(selector.platform, "machine", lambda: "arm64")
        monkeypatch.setattr(selector, "_mlx_preferred", lambda: False)
        assert "mlx-community/" not in _advise(caplog)

    def test_the_off_switch_works(self, apple_silicon_with_mlx, monkeypatch, caplog):
        monkeypatch.setenv("HFL_NO_MLX_HINT", "1")
        assert "mlx-community/" not in _advise(caplog)


class TestItDoesNotOverclaim:
    def test_it_never_asserts_the_mlx_build_exists(self, apple_silicon_with_mlx, caplog):
        """It has not checked, so it must not say it has.

        Guarding the wording, not the mechanism: an advisory that turns
        out to be wrong half the time teaches the user to ignore it, and
        then the real ones go unread too.
        """
        text = _advise(caplog).lower()
        for overclaim in ("an mlx build exists", "there is an mlx build", "is available at"):
            assert overclaim not in text
        assert "typically" in text or "may" in text

    def test_it_does_not_touch_the_network(self, apple_silicon_with_mlx, monkeypatch, caplog):
        """Loading a local model must work with no connection at all."""
        import httpx

        def _boom(*args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("the MLX advisory reached for the network")

        monkeypatch.setattr(httpx, "get", _boom)
        monkeypatch.setattr(httpx, "post", _boom)
        monkeypatch.setattr(httpx.Client, "request", _boom)
        assert "mlx-community/" in _advise(caplog)


def test_the_gguf_branch_still_routes_to_llama_cpp(monkeypatch):
    """The advisory must not change where the model actually goes."""
    from hfl.converter.formats import ModelFormat

    monkeypatch.setattr(selector, "detect_format", lambda p: ModelFormat.GGUF)
    monkeypatch.setattr(selector, "_get_llama_cpp_engine", lambda: "llama-cpp-sentinel")
    monkeypatch.setattr(selector.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(selector.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(selector, "_mlx_preferred", lambda: True)

    assert selector.select_engine("/m/x-Q4_K_M.gguf") == "llama-cpp-sentinel"
