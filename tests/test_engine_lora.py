# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Unit tests for ``hfl/engine/lora.py`` — V4 F4."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hfl.engine.lora import (
    AdapterInfo,
    LoraRegistry,
    apply_lora,
    list_loras,
    remove_lora,
    reset_registry,
)


@pytest.fixture(autouse=True)
def fresh_registry():
    reset_registry()
    yield
    reset_registry()


@pytest.fixture
def adapter_file(tmp_path):
    p = tmp_path / "adapter.safetensors"
    p.write_bytes(b"\x00" * 32)
    return str(p)


def _engine_with_apply():
    """Engine that exposes ``apply_lora`` directly (HFL wrapper style)."""
    engine = MagicMock(spec=["apply_lora", "remove_lora"])
    engine.apply_lora = MagicMock()
    engine.remove_lora = MagicMock()
    return engine


def _engine_via_inner():
    """Engine that proxies to ``_model.set_lora_adapter`` (llama-cpp style)."""

    class _Inner:
        def __init__(self):
            self.set_lora_adapter = MagicMock()
            self.remove_lora_adapter = MagicMock()

    engine = MagicMock(spec=["_model"])
    engine._model = _Inner()
    return engine


def _engine_no_lora():
    """Engine that doesn't support hot-swap at all."""
    return MagicMock(spec=[])


# --- Apply -----------------------------------------------------------------


class TestApplyLora:
    def test_returns_adapter_info(self, adapter_file):
        engine = _engine_with_apply()
        info = apply_lora(engine, lora_path=adapter_file, scale=0.7, name="code")
        assert isinstance(info, AdapterInfo)
        assert info.path == adapter_file
        assert info.scale == 0.7
        assert info.name == "code"
        assert info.adapter_id  # uuid

    def test_engine_apply_lora_is_called(self, adapter_file):
        engine = _engine_with_apply()
        apply_lora(engine, lora_path=adapter_file, scale=0.5)
        engine.apply_lora.assert_called_once_with(adapter_file, 0.5)

    def test_inner_set_lora_adapter_path(self, adapter_file):
        """When the engine doesn't expose ``apply_lora`` directly,
        we fall through to ``_model.set_lora_adapter``."""
        engine = _engine_via_inner()
        apply_lora(engine, lora_path=adapter_file)
        engine._model.set_lora_adapter.assert_called_once()

    def test_unsupported_engine_raises_runtime_error(self, adapter_file):
        with pytest.raises(RuntimeError, match="LoRA hot-swap"):
            apply_lora(_engine_no_lora(), lora_path=adapter_file)

    def test_missing_file_raises(self, tmp_path):
        ghost = tmp_path / "missing.safetensors"
        with pytest.raises(FileNotFoundError):
            apply_lora(_engine_with_apply(), lora_path=str(ghost))

    def test_invalid_scale_raises(self, adapter_file):
        with pytest.raises(ValueError, match="scale"):
            apply_lora(_engine_with_apply(), lora_path=adapter_file, scale=99.0)
        with pytest.raises(ValueError, match="scale"):
            apply_lora(_engine_with_apply(), lora_path=adapter_file, scale=-1.0)


# --- Remove -----------------------------------------------------------------


class TestRemoveLora:
    def test_removes_known_adapter(self, adapter_file):
        engine = _engine_with_apply()
        info = apply_lora(engine, lora_path=adapter_file)

        ok = remove_lora(engine, info.adapter_id)
        assert ok is True
        engine.remove_lora.assert_called_once_with(info.adapter_id)

    def test_unknown_adapter_returns_false(self):
        engine = _engine_with_apply()
        ok = remove_lora(engine, "never-existed")
        assert ok is False
        engine.remove_lora.assert_not_called()

    def test_engine_without_remove_raises(self, adapter_file):
        engine = _engine_with_apply()
        info = apply_lora(engine, lora_path=adapter_file)

        # Strip the remove method.
        broken = MagicMock(spec=[])
        with pytest.raises(RuntimeError, match="removal"):
            remove_lora(broken, info.adapter_id)


# --- Listing ---------------------------------------------------------------


class TestListLoras:
    def test_lists_active_adapters(self, adapter_file):
        engine = _engine_with_apply()
        a = apply_lora(engine, lora_path=adapter_file, scale=0.5, name="A")
        b = apply_lora(engine, lora_path=adapter_file, scale=0.3, name="B")

        adapters = list_loras(engine)
        ids = {x.adapter_id for x in adapters}
        assert {a.adapter_id, b.adapter_id} == ids

    def test_partitioned_by_engine(self, adapter_file):
        e1 = _engine_with_apply()
        e2 = _engine_with_apply()
        apply_lora(e1, lora_path=adapter_file)
        apply_lora(e2, lora_path=adapter_file)

        # Each engine sees only its own adapter.
        assert len(list_loras(e1)) == 1
        assert len(list_loras(e2)) == 1
        # Global view sees both.
        assert len(list_loras()) == 2


# --- Registry plumbing ------------------------------------------------------


class TestLoraRegistry:
    def test_thread_safe_add_and_remove(self):
        reg = LoraRegistry()
        info = AdapterInfo(
            adapter_id="x",
            path="/tmp/x",
            name=None,
            scale=1.0,
            engine_id="e1",
        )
        reg.add(info)
        assert reg.list() == [info]
        assert reg.remove("x") == info
        assert reg.list() == []

    def test_clear(self):
        reg = LoraRegistry()
        reg.add(AdapterInfo(adapter_id="x", path="/tmp/x", name=None, scale=1.0, engine_id="e"))
        reg.clear()
        assert reg.list() == []
