# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for GGUF conversion caching with locking."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from hfl.converter.gguf_converter import (
    GGUFConverter,
    _conversion_locks,
    convert_with_cache,
)


@pytest.fixture(autouse=True)
def _clear_locks():
    """Clear conversion locks between tests."""
    _conversion_locks.clear()
    yield
    _conversion_locks.clear()


@pytest.fixture
def mock_converter():
    """Create a mock GGUFConverter."""
    converter = MagicMock(spec=GGUFConverter)
    return converter


@pytest.fixture
def tmp_paths(tmp_path):
    """Create temporary model and output paths."""
    model_path = tmp_path / "model"
    model_path.mkdir()
    output_path = tmp_path / "output" / "model"
    output_path.parent.mkdir(parents=True)
    return model_path, output_path


class TestConvertWithCacheFastPath:
    """Tests for convert_with_cache when cached file exists."""

    def test_returns_cached_q4_k_m(self, mock_converter, tmp_paths):
        model_path, output_path = tmp_paths
        expected = output_path.with_suffix(".Q4_K_M.gguf")
        expected.write_bytes(b"fake gguf data")

        result = convert_with_cache(mock_converter, model_path, output_path, "Q4_K_M")

        assert result == expected
        mock_converter.convert.assert_not_called()

    def test_returns_cached_f16(self, mock_converter, tmp_paths):
        model_path, output_path = tmp_paths
        expected = output_path.with_suffix(".f16.gguf")
        expected.write_bytes(b"fake gguf data")

        result = convert_with_cache(mock_converter, model_path, output_path, "F16")

        assert result == expected
        mock_converter.convert.assert_not_called()

    def test_returns_cached_q8_0(self, mock_converter, tmp_paths):
        model_path, output_path = tmp_paths
        expected = output_path.with_suffix(".Q8_0.gguf")
        expected.write_bytes(b"fake gguf data")

        result = convert_with_cache(mock_converter, model_path, output_path, "Q8_0")

        assert result == expected
        mock_converter.convert.assert_not_called()

    def test_cached_case_insensitive_quantization(self, mock_converter, tmp_paths):
        model_path, output_path = tmp_paths
        expected = output_path.with_suffix(".Q5_K_M.gguf")
        expected.write_bytes(b"fake gguf data")

        result = convert_with_cache(mock_converter, model_path, output_path, "q5_k_m")

        assert result == expected
        mock_converter.convert.assert_not_called()


class TestConvertWithCacheMiss:
    """Tests for convert_with_cache when no cached file exists."""

    def test_calls_convert_when_no_cache(self, mock_converter, tmp_paths):
        model_path, output_path = tmp_paths
        expected = output_path.with_suffix(".Q4_K_M.gguf")
        mock_converter.convert.return_value = expected

        result = convert_with_cache(mock_converter, model_path, output_path, "Q4_K_M")

        assert result == expected
        mock_converter.convert.assert_called_once_with(model_path, output_path, "Q4_K_M")

    def test_passes_kwargs_to_convert(self, mock_converter, tmp_paths):
        model_path, output_path = tmp_paths
        expected = output_path.with_suffix(".Q4_K_M.gguf")
        mock_converter.convert.return_value = expected

        convert_with_cache(
            mock_converter,
            model_path,
            output_path,
            "Q4_K_M",
            source_repo="org/model",
            original_license="MIT",
        )

        mock_converter.convert.assert_called_once_with(
            model_path,
            output_path,
            "Q4_K_M",
            source_repo="org/model",
            original_license="MIT",
        )


class TestConvertWithCacheDifferentQuantizations:
    """Tests that different quantization levels produce different cache keys."""

    def test_different_quant_levels_are_independent(self, mock_converter, tmp_paths):
        model_path, output_path = tmp_paths

        # Cache Q4_K_M
        q4_output = output_path.with_suffix(".Q4_K_M.gguf")
        q4_output.write_bytes(b"q4 data")

        # Q4_K_M should be cached
        result_q4 = convert_with_cache(mock_converter, model_path, output_path, "Q4_K_M")
        assert result_q4 == q4_output
        mock_converter.convert.assert_not_called()

        # Q8_0 should NOT be cached - needs conversion
        q8_output = output_path.with_suffix(".Q8_0.gguf")
        mock_converter.convert.return_value = q8_output

        result_q8 = convert_with_cache(mock_converter, model_path, output_path, "Q8_0")
        assert result_q8 == q8_output
        mock_converter.convert.assert_called_once()


class TestConvertWithCacheConcurrency:
    """Tests for concurrent access to convert_with_cache."""

    def test_concurrent_calls_same_key_only_convert_once(self, tmp_paths):
        model_path, output_path = tmp_paths
        expected = output_path.with_suffix(".Q4_K_M.gguf")

        call_count = 0
        call_count_lock = threading.Lock()

        def slow_convert(mp, op, q, **kw):
            nonlocal call_count
            # Simulate slow conversion
            time.sleep(0.1)
            # Create the output file to simulate conversion
            expected.write_bytes(b"converted data")
            with call_count_lock:
                call_count += 1
            return expected

        converter = MagicMock(spec=GGUFConverter)
        converter.convert.side_effect = slow_convert

        results = [None, None, None]
        errors = [None, None, None]

        def thread_fn(idx):
            try:
                results[idx] = convert_with_cache(converter, model_path, output_path, "Q4_K_M")
            except Exception as e:
                errors[idx] = e

        threads = [threading.Thread(target=thread_fn, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # No errors
        assert all(e is None for e in errors), f"Errors occurred: {errors}"
        # All got the same result
        assert all(r == expected for r in results)
        # convert was called exactly once
        assert call_count == 1

    def test_concurrent_different_keys_convert_independently(self, tmp_paths):
        model_path, output_path = tmp_paths

        call_keys = []
        call_keys_lock = threading.Lock()

        def tracking_convert(mp, op, q, **kw):
            time.sleep(0.05)
            out = op.with_suffix(f".{q.upper()}.gguf")
            out.write_bytes(b"data")
            with call_keys_lock:
                call_keys.append(q.upper())
            return out

        converter = MagicMock(spec=GGUFConverter)
        converter.convert.side_effect = tracking_convert

        results = [None, None]
        errors = [None, None]

        def thread_fn(idx, quant):
            try:
                results[idx] = convert_with_cache(converter, model_path, output_path, quant)
            except Exception as e:
                errors[idx] = e

        t1 = threading.Thread(target=thread_fn, args=(0, "Q4_K_M"))
        t2 = threading.Thread(target=thread_fn, args=(1, "Q8_0"))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert all(e is None for e in errors), f"Errors: {errors}"
        assert set(call_keys) == {"Q4_K_M", "Q8_0"}


class TestConvertResumeSupport:
    """Tests for resume support in the convert method."""

    @patch("hfl.converter.gguf_converter.subprocess.run")
    def test_skips_fp16_when_intermediate_exists(self, mock_run, tmp_path):
        model_path = tmp_path / "model"
        model_path.mkdir()
        output_path = tmp_path / "output"

        # Create FP16 intermediate file
        fp16_path = output_path.with_suffix(".fp16.gguf")
        fp16_path.write_bytes(b"x" * 1024)

        # Create final output so _verify_output doesn't fail
        final_path = output_path.with_suffix(".Q4_K_M.gguf")

        converter = GGUFConverter()
        converter.ensure_tools = MagicMock()
        converter._verify_output = MagicMock()

        # Make subprocess.run create the final file for quantize step
        def side_effect(cmd, **kwargs):
            if "llama-quantize" in str(cmd[0]) or "quantize" in str(cmd[0]):
                final_path.write_bytes(b"quantized data")
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        converter.convert(model_path, output_path, "Q4_K_M")

        # subprocess.run should have been called only once (quantize step)
        # NOT for the FP16 conversion step
        assert mock_run.call_count == 1
        # The single call should be for quantization
        call_args = mock_run.call_args[0][0]
        assert "quantize" in str(call_args[0]).lower() or "Q4_K_M" in str(call_args)

    @patch("hfl.converter.gguf_converter.subprocess.run")
    def test_runs_fp16_when_no_intermediate(self, mock_run, tmp_path):
        model_path = tmp_path / "model"
        model_path.mkdir()
        output_path = tmp_path / "output"

        fp16_path = output_path.with_suffix(".fp16.gguf")
        final_path = output_path.with_suffix(".Q4_K_M.gguf")

        converter = GGUFConverter()
        converter.ensure_tools = MagicMock()
        converter._verify_output = MagicMock()

        # The converter now runs an env-probe subprocess
        # (``[sys.executable, "-c", ...]``) before the FP16 step. Skip
        # those when classifying call types and only count the real
        # conversion subprocesses.
        convert_step = 0

        def side_effect(cmd, **kwargs):
            nonlocal convert_step
            if isinstance(cmd, list) and len(cmd) > 1 and cmd[1] == "-c":
                # env probe — return success without doing anything
                return MagicMock(returncode=0)
            convert_step += 1
            if convert_step == 1:
                fp16_path.write_bytes(b"fp16 data")
            elif convert_step == 2:
                final_path.write_bytes(b"quantized data")
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        converter.convert(model_path, output_path, "Q4_K_M")

        # Both FP16 conversion and quantization should run
        assert convert_step == 2

    @patch("hfl.converter.gguf_converter.subprocess.run")
    def test_runs_fp16_when_intermediate_is_empty(self, mock_run, tmp_path):
        model_path = tmp_path / "model"
        model_path.mkdir()
        output_path = tmp_path / "output"

        # Create empty FP16 file (interrupted conversion)
        fp16_path = output_path.with_suffix(".fp16.gguf")
        fp16_path.write_bytes(b"")

        final_path = output_path.with_suffix(".Q4_K_M.gguf")

        converter = GGUFConverter()
        converter.ensure_tools = MagicMock()
        converter._verify_output = MagicMock()

        convert_step = 0

        def side_effect(cmd, **kwargs):
            nonlocal convert_step
            if isinstance(cmd, list) and len(cmd) > 1 and cmd[1] == "-c":
                # env probe — return success without doing anything
                return MagicMock(returncode=0)
            convert_step += 1
            if convert_step == 1:
                fp16_path.write_bytes(b"fp16 data")
            elif convert_step == 2:
                final_path.write_bytes(b"quantized data")
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        converter.convert(model_path, output_path, "Q4_K_M")

        # Both steps should run since intermediate was empty
        assert convert_step == 2

    @patch("hfl.converter.gguf_converter.subprocess.run")
    def test_f16_quantization_with_existing_intermediate(self, mock_run, tmp_path):
        """When F16 is requested and fp16 intermediate exists, skip conversion."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        output_path = tmp_path / "output"

        fp16_path = output_path.with_suffix(".fp16.gguf")
        fp16_path.write_bytes(b"fp16 data already here")

        final_path = output_path.with_suffix(".f16.gguf")

        converter = GGUFConverter()
        converter.ensure_tools = MagicMock()
        converter._verify_output = MagicMock()

        result = converter.convert(model_path, output_path, "F16")

        # No subprocess calls needed - fp16 exists, just rename
        mock_run.assert_not_called()
        assert result == final_path


class TestConvertWithCacheLockCreation:
    """Tests for lock management in convert_with_cache."""

    def test_creates_lock_for_new_key(self, mock_converter, tmp_paths):
        model_path, output_path = tmp_paths
        expected = output_path.with_suffix(".Q4_K_M.gguf")
        mock_converter.convert.return_value = expected

        convert_with_cache(mock_converter, model_path, output_path, "Q4_K_M")

        cache_key = f"{model_path}:Q4_K_M"
        assert cache_key in _conversion_locks
        assert isinstance(_conversion_locks[cache_key], type(threading.Lock()))

    def test_reuses_lock_for_same_key(self, mock_converter, tmp_paths):
        model_path, output_path = tmp_paths
        expected = output_path.with_suffix(".Q4_K_M.gguf")

        # First call creates lock
        mock_converter.convert.return_value = expected
        convert_with_cache(mock_converter, model_path, output_path, "Q4_K_M")

        cache_key = f"{model_path}:Q4_K_M"
        lock_after_first = _conversion_locks[cache_key]

        # Second call (file now exists from mock) - create the file
        expected.write_bytes(b"data")
        convert_with_cache(mock_converter, model_path, output_path, "Q4_K_M")

        # Same lock object
        assert _conversion_locks[cache_key] is lock_after_first
