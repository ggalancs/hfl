# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""torch is imported before a CUDA llama.cpp, and only then.

On an L4 with system NCCL 2.25.1, ``import llama_cpp, torch`` failed
(``undefined symbol: ncclCommResume``) and ``import torch, llama_cpp``
worked. Each case runs in its own interpreter with fake packages that
write down the order they were imported in."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def _packages(root: Path, *, cuda: bool, torch: bool) -> Path:
    order = root / "order.txt"
    llama = root / "llama_cpp"
    llama.mkdir()
    (llama / "__init__.py").write_text(f"open({str(order)!r}, 'a').write('llama_cpp\\n')\n")
    (llama / "lib").mkdir()
    if cuda:
        (llama / "lib" / "libggml-cuda.so").write_bytes(b"")
    if torch:
        (root / "torch").mkdir()
        (root / "torch" / "__init__.py").write_text(
            f"open({str(order)!r}, 'a').write('torch\\n')\n"
        )
    return order


def _imported(root: Path, platform: str) -> list[str]:
    code = (
        f"import sys; sys.path[:0] = [{str(root)!r}, {str(SRC)!r}]; "
        f"sys.platform = {platform!r}; import hfl; import llama_cpp"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
    order = root / "order.txt"
    return order.read_text().split() if order.exists() else []


def test_linux_cuda_build_with_torch_installed_imports_torch_first(tmp_path: Path) -> None:
    _packages(tmp_path, cuda=True, torch=True)
    assert _imported(tmp_path, "linux") == ["torch", "llama_cpp"]


@pytest.mark.parametrize(
    ("cuda", "torch", "platform"),
    [
        (False, True, "linux"),  # a CPU build: no NCCL linked, torch not loaded for nothing
        (True, True, "darwin"),  # not Linux
        (True, False, "linux"),  # no torch installed
    ],
)
def test_otherwise_nothing_else_is_imported(
    tmp_path: Path, cuda: bool, torch: bool, platform: str
) -> None:
    _packages(tmp_path, cuda=cuda, torch=torch)
    assert _imported(tmp_path, platform) == ["llama_cpp"]


def test_importing_hfl_imports_neither(tmp_path: Path) -> None:
    _packages(tmp_path, cuda=True, torch=True)
    code = f"import sys; sys.path[:0] = [{str(tmp_path)!r}, {str(SRC)!r}]; import hfl"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
    assert not (tmp_path / "order.txt").exists()
