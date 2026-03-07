# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for memory tracking module."""

from unittest.mock import MagicMock, patch

import pytest

from hfl.engine.memory import (
    MemorySnapshot,
    MemoryTracker,
    check_memory_available,
    format_memory_info,
    get_available_memory,
    get_memory_snapshot,
    measure_memory_delta,
)


class TestMemorySnapshot:
    """Tests for MemorySnapshot dataclass."""

    def test_system_percent(self):
        """Should calculate system memory percentage."""
        snapshot = MemorySnapshot(
            system_used_gb=8.0,
            system_available_gb=8.0,
            system_total_gb=16.0,
        )
        assert snapshot.system_percent == 50.0

    def test_system_percent_zero_total(self):
        """Should handle zero total memory."""
        snapshot = MemorySnapshot(
            system_used_gb=0.0,
            system_available_gb=0.0,
            system_total_gb=0.0,
        )
        assert snapshot.system_percent == 0.0

    def test_gpu_percent(self):
        """Should calculate GPU memory percentage."""
        snapshot = MemorySnapshot(
            system_used_gb=8.0,
            system_available_gb=8.0,
            system_total_gb=16.0,
            gpu_used_gb=4.0,
            gpu_available_gb=4.0,
            gpu_total_gb=8.0,
        )
        assert snapshot.gpu_percent == 50.0

    def test_gpu_percent_no_gpu(self):
        """Should return None when no GPU."""
        snapshot = MemorySnapshot(
            system_used_gb=8.0,
            system_available_gb=8.0,
            system_total_gb=16.0,
        )
        assert snapshot.gpu_percent is None


class TestGetMemorySnapshot:
    """Tests for get_memory_snapshot function."""

    def test_returns_snapshot(self):
        """Should return MemorySnapshot."""
        snapshot = get_memory_snapshot()
        assert isinstance(snapshot, MemorySnapshot)

    def test_with_psutil(self):
        """Should populate system memory when psutil available."""
        import hfl.engine.memory as mem_module

        mock_mem = MagicMock()
        mock_mem.used = 8 * (1024**3)  # 8 GB
        mock_mem.available = 8 * (1024**3)
        mock_mem.total = 16 * (1024**3)

        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = mock_mem

        with patch.object(mem_module, "HAS_PSUTIL", True):
            with patch.object(mem_module, "psutil", mock_psutil, create=True):
                snapshot = get_memory_snapshot()

        assert snapshot.system_used_gb == pytest.approx(8.0, rel=0.01)
        assert snapshot.system_total_gb == pytest.approx(16.0, rel=0.01)

    def test_without_psutil(self):
        """Should return zeros when psutil not available."""
        with patch("hfl.engine.memory.HAS_PSUTIL", False):
            snapshot = get_memory_snapshot()

        assert snapshot.system_used_gb == 0.0
        assert snapshot.system_available_gb == 0.0
        assert snapshot.system_total_gb == 0.0

    def test_without_gputil(self):
        """Should return None for GPU when GPUtil not available."""
        with patch("hfl.engine.memory.HAS_GPUTIL", False):
            snapshot = get_memory_snapshot()

        assert snapshot.gpu_used_gb is None
        assert snapshot.gpu_available_gb is None
        assert snapshot.gpu_total_gb is None


class TestMeasureMemoryDelta:
    """Tests for measure_memory_delta function."""

    def test_positive_delta(self):
        """Should calculate positive memory increase."""
        before = MemorySnapshot(
            system_used_gb=4.0,
            system_available_gb=12.0,
            system_total_gb=16.0,
        )
        after = MemorySnapshot(
            system_used_gb=8.0,
            system_available_gb=8.0,
            system_total_gb=16.0,
        )

        delta = measure_memory_delta(before, after)
        assert delta == 4.0

    def test_negative_delta_returns_zero(self):
        """Should return zero for negative delta (memory freed)."""
        before = MemorySnapshot(
            system_used_gb=8.0,
            system_available_gb=8.0,
            system_total_gb=16.0,
        )
        after = MemorySnapshot(
            system_used_gb=4.0,
            system_available_gb=12.0,
            system_total_gb=16.0,
        )

        delta = measure_memory_delta(before, after)
        assert delta == 0.0

    def test_with_gpu_delta(self):
        """Should include GPU memory in delta."""
        before = MemorySnapshot(
            system_used_gb=4.0,
            system_available_gb=12.0,
            system_total_gb=16.0,
            gpu_used_gb=2.0,
            gpu_available_gb=6.0,
            gpu_total_gb=8.0,
        )
        after = MemorySnapshot(
            system_used_gb=5.0,
            system_available_gb=11.0,
            system_total_gb=16.0,
            gpu_used_gb=4.0,
            gpu_available_gb=4.0,
            gpu_total_gb=8.0,
        )

        delta = measure_memory_delta(before, after)
        assert delta == 3.0  # 1 GB system + 2 GB GPU


class TestGetAvailableMemory:
    """Tests for get_available_memory function."""

    def test_returns_tuple(self):
        """Should return tuple of system and GPU memory."""
        system, gpu = get_available_memory()
        assert isinstance(system, float)
        assert gpu is None or isinstance(gpu, float)


class TestCheckMemoryAvailable:
    """Tests for check_memory_available function."""

    def test_enough_system_memory(self):
        """Should return True when enough system memory."""
        mock_snapshot = MemorySnapshot(
            system_used_gb=4.0,
            system_available_gb=12.0,
            system_total_gb=16.0,
        )

        with patch("hfl.engine.memory.get_memory_snapshot", return_value=mock_snapshot):
            result = check_memory_available(8.0)

        assert result is True

    def test_not_enough_memory(self):
        """Should return False when not enough memory."""
        mock_snapshot = MemorySnapshot(
            system_used_gb=14.0,
            system_available_gb=2.0,
            system_total_gb=16.0,
        )

        with patch("hfl.engine.memory.get_memory_snapshot", return_value=mock_snapshot):
            result = check_memory_available(8.0)

        assert result is False

    def test_prefer_gpu(self):
        """Should check GPU first when prefer_gpu=True."""
        mock_snapshot = MemorySnapshot(
            system_used_gb=14.0,
            system_available_gb=2.0,
            system_total_gb=16.0,
            gpu_used_gb=2.0,
            gpu_available_gb=6.0,
            gpu_total_gb=8.0,
        )

        with patch("hfl.engine.memory.get_memory_snapshot", return_value=mock_snapshot):
            result = check_memory_available(4.0, prefer_gpu=True)

        assert result is True  # GPU has enough even though system doesn't


class TestFormatMemoryInfo:
    """Tests for format_memory_info function."""

    def test_format_without_gpu(self):
        """Should format system memory only when no GPU."""
        mock_snapshot = MemorySnapshot(
            system_used_gb=8.0,
            system_available_gb=8.0,
            system_total_gb=16.0,
        )

        with patch("hfl.engine.memory.get_memory_snapshot", return_value=mock_snapshot):
            result = format_memory_info()

        assert "System RAM:" in result
        assert "8.0/16.0 GB" in result
        assert "GPU: Not available" in result

    def test_format_with_gpu(self):
        """Should format both system and GPU memory."""
        mock_snapshot = MemorySnapshot(
            system_used_gb=8.0,
            system_available_gb=8.0,
            system_total_gb=16.0,
            gpu_used_gb=4.0,
            gpu_available_gb=4.0,
            gpu_total_gb=8.0,
            gpu_id=0,
        )

        with patch("hfl.engine.memory.get_memory_snapshot", return_value=mock_snapshot):
            result = format_memory_info()

        assert "System RAM:" in result
        assert "GPU 0 VRAM:" in result


class TestMemoryTracker:
    """Tests for MemoryTracker context manager."""

    def test_context_manager(self):
        """Should track memory as context manager."""
        with MemoryTracker() as tracker:
            # Simulate some memory allocation
            _data = [0] * 1000000  # noqa: F841

        assert tracker.before is not None
        assert tracker.after is not None
        assert isinstance(tracker.memory_used_gb, float)

    def test_memory_used_gb(self):
        """Should calculate memory used."""
        tracker = MemoryTracker()
        tracker.before = MemorySnapshot(
            system_used_gb=4.0,
            system_available_gb=12.0,
            system_total_gb=16.0,
        )
        tracker.after = MemorySnapshot(
            system_used_gb=6.0,
            system_available_gb=10.0,
            system_total_gb=16.0,
        )

        assert tracker.memory_used_gb == 2.0

    def test_system_delta_gb(self):
        """Should calculate system memory delta."""
        tracker = MemoryTracker()
        tracker.before = MemorySnapshot(
            system_used_gb=4.0,
            system_available_gb=12.0,
            system_total_gb=16.0,
        )
        tracker.after = MemorySnapshot(
            system_used_gb=7.0,
            system_available_gb=9.0,
            system_total_gb=16.0,
        )

        assert tracker.system_delta_gb == 3.0

    def test_gpu_delta_gb(self):
        """Should calculate GPU memory delta."""
        tracker = MemoryTracker()
        tracker.before = MemorySnapshot(
            system_used_gb=4.0,
            system_available_gb=12.0,
            system_total_gb=16.0,
            gpu_used_gb=2.0,
            gpu_available_gb=6.0,
            gpu_total_gb=8.0,
        )
        tracker.after = MemorySnapshot(
            system_used_gb=4.0,
            system_available_gb=12.0,
            system_total_gb=16.0,
            gpu_used_gb=5.0,
            gpu_available_gb=3.0,
            gpu_total_gb=8.0,
        )

        assert tracker.gpu_delta_gb == 3.0

    def test_gpu_delta_no_gpu(self):
        """Should return None when no GPU."""
        tracker = MemoryTracker()
        tracker.before = MemorySnapshot(
            system_used_gb=4.0,
            system_available_gb=12.0,
            system_total_gb=16.0,
        )
        tracker.after = MemorySnapshot(
            system_used_gb=4.0,
            system_available_gb=12.0,
            system_total_gb=16.0,
        )

        assert tracker.gpu_delta_gb is None

    def test_before_enter(self):
        """Should return zeros before entering context."""
        tracker = MemoryTracker()
        assert tracker.memory_used_gb == 0.0
        assert tracker.system_delta_gb == 0.0
