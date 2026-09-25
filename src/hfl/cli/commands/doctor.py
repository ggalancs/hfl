# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl doctor`` diagnostic (Phase 15 P2 — V2 row 15).

Prints a summary of the detected hardware + software stack so
operators can answer "why isn't my GPU being used?" without
reading the llama-cpp logs.

Checks:
  - Python version + platform.
  - llama-cpp-python installed? Built with CUDA / Metal / Vulkan /
    ROCm?
  - NVIDIA devices visible via pynvml.
  - Metal availability on Darwin-arm64.
  - ROCm cards visible under ``/sys/class/drm``.
  - MLX / transformers / vLLM extras present.
  - Current VRAM probe tier → recommended ``num_ctx``.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DoctorReport", "build_report", "format_report"]


@dataclass
class DoctorReport:
    """A single-shot snapshot of the runtime environment."""

    python_version: str = ""
    platform_system: str = ""
    platform_machine: str = ""

    llama_cpp_available: bool = False
    llama_cpp_build_features: dict[str, bool] = field(default_factory=dict)
    # llama.cpp's ``llama-server``: what ``hfl serve --parallel`` runs.
    llama_server: str | None = None

    nvidia_devices: list[str] = field(default_factory=list)
    metal_available: bool = False
    rocm_devices: list[str] = field(default_factory=list)

    mlx_available: bool = False
    transformers_available: bool = False
    vllm_available: bool = False

    vram_gib: float | None = None
    recommended_ctx: int = 0

    # macOS laptops only: "battery" / "AC power" / "unknown". Running on
    # battery costs 30-50% of throughput with every layer still on the GPU,
    # so it belongs in the diagnostic next to the accelerator list.
    power_source: str = "unknown"

    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


# ----------------------------------------------------------------------
# Probes
# ----------------------------------------------------------------------


def _probe_llama_cpp() -> tuple[bool, dict[str, bool]]:
    try:
        import llama_cpp
    except ImportError:
        return False, {}
    features: dict[str, bool] = {}
    # llama-cpp-python exposes a ``llama_supports_gpu_offload`` helper
    # and backend-specific flags via ``ggml.supports_*``. Both are
    # best-effort; wrapping in try/except keeps the probe resilient.
    from hfl.engine.llama_cpp import _suppress_stderr

    for name in ("supports_gpu_offload", "llama_supports_gpu_offload"):
        fn = getattr(llama_cpp, name, None)
        if callable(fn):
            # Asking initialises the GPU backend, which prints ~20 lines of
            # ggml_metal_* setup straight to fd 2 before the report.
            try:
                with _suppress_stderr():
                    features["gpu_offload"] = bool(fn())
            except Exception:  # pragma: no cover
                features["gpu_offload"] = False
            break
    return True, features


def _probe_nvidia() -> list[str]:
    try:
        import pynvml
    except ImportError:
        return _probe_nvidia_torch()
    try:
        pynvml.nvmlInit()
    except Exception:
        return []
    out: list[str] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            out.append(name)
    except Exception:
        pass
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return out


def _probe_nvidia_torch() -> list[str]:
    """NVIDIA devices seen by PyTorch, when pynvml is not installed."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def _probe_metal() -> bool:
    if platform.system() != "Darwin":
        return False
    if platform.machine().lower() not in ("arm64", "aarch64"):
        return False
    try:
        import torch

        if not getattr(torch.backends, "mps", None):
            return True  # Darwin-arm64 alone implies Metal.
        return bool(torch.backends.mps.is_available())
    except ImportError:
        # torch-less installs still have Metal; probe cleanly and say yes.
        return True


def _probe_rocm() -> list[str]:
    from pathlib import Path

    root = Path("/sys/class/drm")
    if not root.exists():
        return []
    cards: list[str] = []
    try:
        for entry in sorted(root.iterdir()):
            if entry.name.startswith("card") and entry.name[4:].isdigit():
                # We can read the card's PCI device id as a simple tag.
                tag = entry.name
                pci = entry / "device" / "device"
                if pci.exists():
                    try:
                        tag = f"{entry.name} ({pci.read_text().strip()})"
                    except OSError:
                        pass
                cards.append(tag)
    except Exception:  # pragma: no cover
        pass
    return cards


def _probe_optional(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


# ----------------------------------------------------------------------
# Report assembly
# ----------------------------------------------------------------------


def build_report() -> DoctorReport:
    report = DoctorReport(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform_system=platform.system(),
        platform_machine=platform.machine(),
    )

    report.llama_cpp_available, report.llama_cpp_build_features = _probe_llama_cpp()
    report.nvidia_devices = _probe_nvidia()
    report.metal_available = _probe_metal()
    report.rocm_devices = _probe_rocm()
    report.mlx_available = _probe_optional("mlx_lm")
    report.transformers_available = _probe_optional("transformers")
    report.vllm_available = _probe_optional("vllm")
    try:
        from hfl.engine.llama_server import binary

        report.llama_server = binary()
    except Exception:  # pragma: no cover — advisory only
        report.llama_server = None

    try:
        from hfl.engine.vram import pick_ctx_size

        tier = pick_ctx_size()
        report.vram_gib = tier.vram_gib
        report.recommended_ctx = tier.ctx
    except Exception:  # pragma: no cover
        report.recommended_ctx = 4096

    try:
        from hfl.engine.power import power_source_label

        report.power_source = power_source_label()
    except Exception:  # pragma: no cover — advisory only
        report.power_source = "unknown"

    # Synthesised recommendations.
    recs: list[str] = []
    if report.power_source == "battery":
        recs.append(
            "Running on battery — macOS clocks the GPU down hard (measured 8x "
            "slower generation on an M3 Max). Plug in the charger."
        )
    if not report.llama_cpp_available:
        recs.append("llama-cpp-python missing. Install with `pip install 'hfl[llama]'`.")
    if report.llama_server is None:
        recs.append(
            "llama.cpp's llama-server not found: GGUF models answer one request at a "
            "time. Install llama.cpp (e.g. `brew install llama.cpp`) for "
            "`hfl serve --parallel N`."
        )
    has_accel = bool(report.nvidia_devices) or report.metal_available or bool(report.rocm_devices)
    if not has_accel:
        recs.append(
            "No GPU accelerator detected. Expect CPU-only inference; "
            "consider a GGUF Q4_K_M quant for a 7B model."
        )
    if (
        report.platform_system == "Darwin"
        and platform.machine() == "arm64"
        and not report.mlx_available
    ):
        recs.append("On Apple Silicon — `pip install 'hfl[mlx]'` unlocks the MLX backend.")
    report.recommendations = recs
    return report


# ----------------------------------------------------------------------
# Shared rendering — `hfl check`, `hfl debug` and `hfl doctor` all read the
# backends and accelerators from here, so they can no longer disagree.
# ----------------------------------------------------------------------


def backend_rows(report: DoctorReport) -> list[tuple[str, bool, str]]:
    """(name, available, detail) for every inference backend."""
    offload = report.llama_cpp_build_features.get("gpu_offload")
    return [
        (
            "llama-cpp",
            report.llama_cpp_available,
            "" if offload is None else f"GPU offload {'✓' if offload else '✗'}",
        ),
        (
            "llama-server",
            report.llama_server is not None,
            report.llama_server or "needed for `hfl serve --parallel`",
        ),
        ("transformers", report.transformers_available, ""),
        ("vllm", report.vllm_available, ""),
        ("mlx", report.mlx_available, ""),
    ]


def accelerator_rows(report: DoctorReport) -> list[tuple[str, str]]:
    """(label, detail) for every accelerator found; CPU when there is none."""
    rows = [("NVIDIA", name) for name in report.nvidia_devices]
    if report.metal_available:
        rows.append(("Apple Metal", ""))
    rows += [("AMD ROCm", card) for card in report.rocm_devices]
    return rows or [("CPU only", "no GPU accelerator found")]


def format_report(report: DoctorReport) -> str:
    def _yn(value: bool) -> str:
        return "✓" if value else "✗"

    lines: list[str] = []
    lines.append("hfl doctor — runtime diagnostic")
    lines.append("")
    py = report.python_version
    plat = f"{report.platform_system} {report.platform_machine}"
    lines.append(f"Python:           {py} ({plat})")
    lines.append("")
    lines.append("Backends:")
    for name, ok, detail in backend_rows(report):
        lines.append(f"  {name:<15} {_yn(ok)}" + (f"  {detail}" if detail else ""))
    lines.append("")
    lines.append("Accelerators:")
    for label, detail in accelerator_rows(report):
        lines.append(f"  {label:<15} " + (detail or "✓"))
    lines.append("")
    vram_text = f"{report.vram_gib:.1f} GiB" if report.vram_gib is not None else "unknown"
    ctx_rec = report.recommended_ctx
    lines.append(f"Detected VRAM:    {vram_text} → num_ctx recommendation: {ctx_rec}")
    if report.power_source != "unknown":
        lines.append(f"Power source:     {report.power_source}")
    if report.recommendations:
        lines.append("")
        lines.append("Recommendations:")
        for rec in report.recommendations:
            lines.append(f"  • {rec}")
    return "\n".join(lines) + "\n"
