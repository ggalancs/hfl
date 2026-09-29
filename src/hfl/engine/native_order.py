# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""torch before llama.cpp's CUDA build, when a process may load both.

A llama-cpp-python built with CUDA on a machine that has NCCL installed
links the system ``libnccl.so.2``. Loaded first, it is the NCCL torch binds
to as well, and a torch that ships a newer one then fails to import:
``libtorch_cuda.so: undefined symbol: ncclCommResume`` (measured on an L4:
system NCCL 2.25.1, torch 2.13 with its own 2.29.7). Transformers, vLLM
and embeddings all broke once a GGUF model had been loaded; torch first,
both load. So the first import of ``llama_cpp`` imports torch before it —
only on Linux, only for a CUDA build of llama.cpp, only when torch is
installed. Elsewhere nothing is imported and nothing changes.
"""

from __future__ import annotations

import sys


def _torch_goes_first() -> bool:
    import importlib.util
    from pathlib import Path

    if not sys.platform.startswith("linux") or "torch" in sys.modules:
        return False
    try:
        if importlib.util.find_spec("torch") is None:
            return False
        spec = importlib.util.find_spec("llama_cpp")
    except (ImportError, ValueError):
        return False
    if spec is None or not spec.submodule_search_locations:
        return False
    lib = Path(list(spec.submodule_search_locations)[0]) / "lib"
    return any(lib.glob("libggml-cuda*"))


class _TorchFirst:
    """Waits for the first import of ``llama_cpp``; imports torch then, if
    it must, and steps aside for the real import."""

    def find_spec(self, fullname: str, path: object, target: object = None) -> None:
        if fullname != "llama_cpp":
            return None
        # Once: out of the way before anything below imports.
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        if _torch_goes_first():
            try:
                import torch  # noqa: F401
            except Exception:  # a broken torch must not stop llama.cpp
                pass
        return None


def install_torch_first() -> None:
    """Idempotent; costs nothing until ``llama_cpp`` is imported."""
    if "llama_cpp" in sys.modules:
        return
    if not any(isinstance(f, _TorchFirst) for f in sys.meta_path):
        # Duck-typed, as ``hfl.hub.timeouts``: no ``importlib`` imports here.
        sys.meta_path.insert(0, _TorchFirst())
