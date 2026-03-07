# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Memory tracking utilities for model management.

Provides real memory measurement for system RAM and GPU VRAM.
Used by ModelPool for accurate memory-based eviction.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# Optional dependencies for memory tracking
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import GPUtil

    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time.

    Attributes:
        system_used_gb: System RAM currently in use (GB)
        system_available_gb: System RAM available (GB)
        system_total_gb: Total system RAM (GB)
        gpu_used_gb: GPU VRAM currently in use (GB), None if no GPU
        gpu_available_gb: GPU VRAM available (GB), None if no GPU
        gpu_total_gb: Total GPU VRAM (GB), None if no GPU
        gpu_id: GPU device ID, None if no GPU
        timestamp: Unix timestamp when snapshot was taken
    """

    system_used_gb: float
    system_available_gb: float
    system_total_gb: float
    gpu_used_gb: float | None = None
    gpu_available_gb: float | None = None
    gpu_total_gb: float | None = None
    gpu_id: int | None = None
    timestamp: float = field(default_factory=time.time)

    @property
    def system_percent(self) -> float:
        """System memory usage as percentage."""
        if self.system_total_gb == 0:
            return 0.0
        return (self.system_used_gb / self.system_total_gb) * 100

    @property
    def gpu_percent(self) -> float | None:
        """GPU memory usage as percentage, None if no GPU."""
        if self.gpu_total_gb is None or self.gpu_total_gb == 0:
            return None
        if self.gpu_used_gb is None:
            return None
        return (self.gpu_used_gb / self.gpu_total_gb) * 100


def get_memory_snapshot(gpu_id: int = 0) -> MemorySnapshot:
    """Get current memory usage snapshot.

    Args:
        gpu_id: GPU device ID to query (default 0)

    Returns:
        MemorySnapshot with current memory state

    Note:
        If psutil is not installed, returns zeros for system memory.
        If GPUtil is not installed or no GPU available, GPU fields are None.
    """
    # System memory
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        system_used = mem.used / (1024**3)
        system_available = mem.available / (1024**3)
        system_total = mem.total / (1024**3)
    else:
        system_used = 0.0
        system_available = 0.0
        system_total = 0.0

    # GPU memory
    gpu_used = None
    gpu_available = None
    gpu_total = None
    actual_gpu_id = None

    if HAS_GPUTIL:
        try:
            gpus = GPUtil.getGPUs()
            if gpus and gpu_id < len(gpus):
                gpu = gpus[gpu_id]
                gpu_used = gpu.memoryUsed / 1024  # MB to GB
                gpu_total = gpu.memoryTotal / 1024
                gpu_available = gpu_total - gpu_used
                actual_gpu_id = gpu_id
        except (OSError, ValueError, RuntimeError, IndexError) as e:
            # GPUtil can fail on some systems (no GPU, driver issues, etc.)
            logger.debug(f"GPU memory query failed: {e}")

    return MemorySnapshot(
        system_used_gb=system_used,
        system_available_gb=system_available,
        system_total_gb=system_total,
        gpu_used_gb=gpu_used,
        gpu_available_gb=gpu_available,
        gpu_total_gb=gpu_total,
        gpu_id=actual_gpu_id,
    )


def measure_memory_delta(before: MemorySnapshot, after: MemorySnapshot) -> float:
    """Calculate memory used between two snapshots.

    Args:
        before: Snapshot before operation
        after: Snapshot after operation

    Returns:
        Total memory increase in GB (system + GPU)
    """
    system_delta = max(0, after.system_used_gb - before.system_used_gb)

    gpu_delta = 0.0
    if after.gpu_used_gb is not None and before.gpu_used_gb is not None:
        gpu_delta = max(0, after.gpu_used_gb - before.gpu_used_gb)

    return system_delta + gpu_delta


def get_available_memory() -> tuple[float, float | None]:
    """Get available memory for model loading.

    Returns:
        Tuple of (system_available_gb, gpu_available_gb)
        GPU value is None if no GPU available.
    """
    snapshot = get_memory_snapshot()
    return snapshot.system_available_gb, snapshot.gpu_available_gb


def check_memory_available(required_gb: float, prefer_gpu: bool = True) -> bool:
    """Check if enough memory is available for model.

    Args:
        required_gb: Memory required in GB
        prefer_gpu: If True, check GPU first if available

    Returns:
        True if enough memory is available
    """
    system_avail, gpu_avail = get_available_memory()

    if prefer_gpu and gpu_avail is not None:
        if gpu_avail >= required_gb:
            return True

    return system_avail >= required_gb


def format_memory_info() -> str:
    """Format current memory state as human-readable string.

    Returns:
        Formatted string showing memory usage
    """
    snapshot = get_memory_snapshot()

    lines = [
        f"System RAM: {snapshot.system_used_gb:.1f}/{snapshot.system_total_gb:.1f} GB "
        f"({snapshot.system_percent:.1f}% used)"
    ]

    if snapshot.gpu_total_gb is not None:
        gpu_pct = snapshot.gpu_percent or 0
        lines.append(
            f"GPU {snapshot.gpu_id} VRAM: {snapshot.gpu_used_gb:.1f}/{snapshot.gpu_total_gb:.1f} GB "
            f"({gpu_pct:.1f}% used)"
        )
    else:
        lines.append("GPU: Not available")

    return "\n".join(lines)


class MemoryTracker:
    """Context manager for tracking memory usage of operations.

    Example:
        with MemoryTracker() as tracker:
            model = load_model()

        print(f"Model uses {tracker.memory_used_gb:.2f} GB")
    """

    def __init__(self, gpu_id: int = 0):
        """Initialize tracker.

        Args:
            gpu_id: GPU device ID to track
        """
        self.gpu_id = gpu_id
        self.before: MemorySnapshot | None = None
        self.after: MemorySnapshot | None = None

    def __enter__(self) -> "MemoryTracker":
        self.before = get_memory_snapshot(self.gpu_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.after = get_memory_snapshot(self.gpu_id)

    @property
    def memory_used_gb(self) -> float:
        """Memory used during tracked operation."""
        if self.before is None or self.after is None:
            return 0.0
        return measure_memory_delta(self.before, self.after)

    @property
    def system_delta_gb(self) -> float:
        """System RAM delta."""
        if self.before is None or self.after is None:
            return 0.0
        return max(0, self.after.system_used_gb - self.before.system_used_gb)

    @property
    def gpu_delta_gb(self) -> float | None:
        """GPU VRAM delta, None if no GPU."""
        if self.before is None or self.after is None:
            return None
        if self.after.gpu_used_gb is None or self.before.gpu_used_gb is None:
            return None
        return max(0, self.after.gpu_used_gb - self.before.gpu_used_gb)
