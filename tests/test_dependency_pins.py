# SPDX-License-Identifier: Apache-2.0
"""Regression guards for cross-package dependency compatibility.

HFL targets the post-2025 Hugging Face stack: ``transformers>=5`` paired
with ``huggingface_hub>=1``. Both majors bumped together and share the
new httpx-based hub client. Mixing the 4.x / 0.x lines with the 5.x /
1.x lines crashes the ``convert_hf_to_gguf.py`` import chain.

This module adds two layers of defence:

1. **Static guard** — parses ``pyproject.toml`` and asserts the pin
   bounds for ``huggingface-hub`` and the ``[transformers]`` /
   ``[tts]`` extras stay on combinations we know are compatible.
   Always runs, no optional deps required, so it fires in default CI.

2. **Smoke import** — when ``transformers`` AND ``huggingface_hub``
   are both importable in the current interpreter, exercise the
   import chain ``convert_hf_to_gguf.py`` walks. Skipped when the
   optional packages aren't installed.

If either layer fires, fix the pin (or the host Python) before
landing the change.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — 3.10 fallback
    import tomli as tomllib  # type: ignore[import-not-found]


PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _load_pyproject() -> dict:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def _find_pin(specs: list[str], package: str) -> str:
    """Return the version specifier for ``package`` from ``specs``."""
    for spec in specs:
        # Match the leading package name (split on ``[``, ``>=``, ``==``, ...).
        head = re.split(r"[\[<>=!~ ]", spec, maxsplit=1)[0].strip()
        if head == package:
            return spec
    raise AssertionError(f"package {package!r} not found in {specs}")


# --- 1. Static pin guards -----------------------------------------------------


class TestStaticPyprojectPins:
    """Static checks that don't need the optional packages installed.

    These run in the default ``[dev]`` CI environment so a regression
    in the pin bounds is caught before merge.
    """

    def test_huggingface_hub_pin_targets_1x(self):
        cfg = _load_pyproject()
        deps = cfg["project"]["dependencies"]
        pin = _find_pin(deps, "huggingface-hub")
        # HFL pairs with the httpx-based hub 1.x generation so
        # transformers 5.x can import is_offline_mode from the
        # canonical location without blowing up.
        assert ">=1.5" in pin, (
            f"huggingface-hub pin must floor at >=1.5 — transformers 5.x's own "
            f"Requires-Dist is huggingface-hub>=1.5.0: {pin!r}\n"
            "Dropping below 1.5 re-introduces the is_offline_mode "
            "ImportError chain that crashed convert_hf_to_gguf.py."
        )
        # <2.0 until tokenizers & co allow 2.x: with <3.0, [stt] resolved
        # tokenizers 0.13.3 (no wheel, needs Rust). See pyproject.
        assert "<2.0" in pin or "<2," in pin, (
            f"huggingface-hub pin must stay below 2.0 until the ecosystem allows it: {pin!r}"
        )

    @pytest.mark.parametrize("extra_name", ["transformers", "tts"])
    def test_transformers_extras_target_5x(self, extra_name):
        cfg = _load_pyproject()
        extras = cfg["project"]["optional-dependencies"][extra_name]
        pin = _find_pin(extras, "transformers")
        # Paired with huggingface_hub 1.x above. Anything below 5.x
        # pulls the old hub 0.x client and breaks the import chain.
        assert ">=5.0" in pin or ">=5," in pin, (
            f"[{extra_name}] transformers pin must target 5.x: {pin!r}\n"
            "Dropping to the 4.x line desynchronises us from hub 1.x."
        )
        assert "<6.0" in pin or "<6," in pin, (
            f"[{extra_name}] transformers pin must cap before the next major: {pin!r}"
        )

    def test_mcp_takes_1x_and_2x_but_not_the_next_major(self):
        """hfl.mcp.server serves both SDK APIs (2.x dropped the decorators
        1.x used, local audit A21). A cap below 2 made `hfl[all]` fall back
        to a vLLM with no wheel, vLLM 0.30 requiring mcp 2 (Linux audit C1)."""
        pin = _find_pin(_load_pyproject()["project"]["optional-dependencies"]["mcp"], "mcp")
        compact = pin.replace(" ", "")
        assert "<3" in compact and "<2" not in compact, pin

    def test_structured_output_comes_with_its_backends(self):
        """llguidance is capped below its next major, and [mlx] and
        [transformers] bring it: without it those backends refuse a response
        format (plan 0.22 P1-8)."""
        extras = _load_pyproject()["project"]["optional-dependencies"]
        pin = _find_pin(extras["structured"], "llguidance")
        assert "<2" in pin.replace(" ", ""), pin
        # vLLM 0.30 pins llguidance <1.8: a 1.8 floor broke `hfl[all]`.
        assert ">=1.7" in pin.replace(" ", ""), pin
        assert any(dep.startswith("hfl[structured]") for dep in extras["transformers"])
        # [mlx] names it with the platform marker (a self-reference loses it
        # in the wheel), at the same range as [structured].
        mlx = [d for d in extras["mlx"] if d.startswith("llguidance")]
        assert len(mlx) == 1 and mlx[0].split(";")[0].strip() == pin.strip(), mlx
        assert "sys_platform == 'darwin'" in mlx[0], mlx

    def test_vllm_is_linux_only_with_a_floor_that_has_wheels(self):
        """A low floor let the resolver fall back to vLLM 0.6.5, which has no
        wheel, whenever another extra wanted newer torch/llguidance:
        `hfl[all]` failed building it on Linux arm64 (Linux audit C1)."""
        deps = _load_pyproject()["project"]["optional-dependencies"]["vllm"]
        assert len(deps) == 1
        spec, _, marker = deps[0].partition(";")
        assert ">=0.30" in spec.replace(" ", "") and "sys_platform == 'linux'" in marker

    def test_coqui_brings_its_codec(self):
        """From torch 2.9 coqui-tts refuses to import without torchcodec,
        which its [codec] extra installs (local audit C5)."""
        pin = _find_pin(_load_pyproject()["project"]["optional-dependencies"]["coqui"], "coqui-tts")
        assert "[codec]" in pin, pin

    def test_train_brings_peft_for_transformers_5(self):
        """`hfl train` outside Apple Silicon trains with PEFT; 0.17+ is what
        works with transformers 5 (resolved with it: peft 0.21.1)."""
        deps = _load_pyproject()["project"]["optional-dependencies"]
        pin = _find_pin(deps["train"], "peft")
        assert ">=0.17" in pin.replace(" ", "") and "<1.0" in pin.replace(" ", ""), pin
        assert "train" in deps["all"][0], deps["all"]

    def test_stt_keeps_a_pyav_faster_whisper_can_call(self):
        """PyAV 19 dropped av.open(metadata_errors=...), which faster-whisper
        1.2.1 passes: every transcription failed (local audit B32/B54/E8)."""
        pin = _find_pin(_load_pyproject()["project"]["optional-dependencies"]["stt"], "av")
        assert "<19" in pin.replace(" ", ""), pin

    def test_llama_cpp_python_pin_supports_gemma4(self):
        """The ``[llama]`` extra must require a llama-cpp-python build
        that recognises the ``gemma4`` GGUF architecture.

        Pre-0.3.20 builds reject Gemma 4 GGUFs with
        ``unknown model architecture: 'gemma4'``. Anything older brings
        back the bug for fresh installs.
        """
        cfg = _load_pyproject()
        extras = cfg["project"]["optional-dependencies"]["llama"]
        pin = _find_pin(extras, "llama-cpp-python")
        # Parse the lower bound
        match = re.search(r">=([\d.]+)", pin)
        assert match, f"llama-cpp-python pin must have a >= bound: {pin!r}"
        major, minor, *patch = match.group(1).split(".")
        bound = (int(major), int(minor), int(patch[0]) if patch else 0)
        assert bound >= (0, 3, 20), (
            f"llama-cpp-python pin {pin!r} is below the Gemma-4 cutoff. Bump to >=0.3.20."
        )


# --- 2. Live smoke import -----------------------------------------------------


_HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
_HAS_HF_HUB = importlib.util.find_spec("huggingface_hub") is not None


@pytest.mark.skipif(
    not (_HAS_TRANSFORMERS and _HAS_HF_HUB),
    reason="optional [transformers] extra not installed",
)
class TestTransformersImportChain:
    """Runs only when both packages are present.

    These tests exercise the same import path that
    ``llama.cpp/convert_hf_to_gguf.py`` walks when HFL invokes it via
    ``sys.executable`` to convert a HuggingFace model to GGUF. If they
    fail, ``hfl pull <model>`` will fail at conversion time with a
    cryptic traceback — fix the host env before that happens in
    production.
    """

    def test_top_level_transformers_imports(self):
        """``import transformers`` must succeed cleanly.

        This catches the historical
        ``ImportError: cannot import name 'is_offline_mode' from
        'huggingface_hub'`` that broke the host Python in April 2026.
        """
        import transformers  # noqa: F401

    def test_auto_classes_import(self):
        """The two classes ``convert_hf_to_gguf.py`` actually uses."""
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

    def test_huggingface_hub_top_level(self):
        """``huggingface_hub`` must import cleanly so HFL's hub client
        and the conversion script can talk to the Hub."""
        import huggingface_hub  # noqa: F401

    def test_versions_are_a_compatible_combo(self):
        """When both packages are installed, they must agree on the
        major version axis. HFL's supported pairing is transformers 5.x
        with huggingface_hub 1.x. The old 4.x / 0.x pairing is no
        longer supported."""
        import huggingface_hub
        import transformers

        tx_major = int(transformers.__version__.split(".")[0])
        hh_major = int(huggingface_hub.__version__.split(".")[0])
        if tx_major == 5:
            assert hh_major == 1, (
                f"transformers 5.x requires huggingface_hub 1.x, got "
                f"{huggingface_hub.__version__}. Run "
                f"``pip install --upgrade --force-reinstall "
                f"'huggingface-hub>=1.0.0,<2.0'``."
            )
        elif tx_major == 4:
            pytest.fail(
                f"transformers 4.x ({transformers.__version__}) is no "
                "longer supported by HFL. Upgrade with "
                "``pip install --upgrade "
                "'transformers>=5.0.0,<6.0' "
                "'huggingface-hub>=1.0.0,<2.0'``."
            )
        else:
            pytest.fail(f"unexpected transformers major: {tx_major}")


def test_all_leaves_llama_cpp_python_out_on_windows() -> None:
    """PyPI has no Windows wheel for llama-cpp-python: `pip install
    "hfl[all]"` failed on a clean Windows 10 without the C++ Build Tools.
    Everywhere else ``all`` still brings it, at [llama]'s own pin.

    Named directly: written as ``"hfl[llama]; <marker>"`` hatchling expanded
    the self-reference and dropped the marker, and the built wheel still
    required llama-cpp-python on Windows (seen in its METADATA)."""
    from packaging.requirements import Requirement

    deps = _load_pyproject()["project"]["optional-dependencies"]
    entries = [Requirement(r) for r in deps["all"]]
    assert not any(r.name == "hfl" and "llama" in r.extras for r in entries), deps["all"]
    llama = [r for r in entries if r.name == "llama-cpp-python"]
    assert len(llama) == 1 and llama[0].marker is not None, deps["all"]
    marker = llama[0].marker
    assert not marker.evaluate({"sys_platform": "win32", "extra": ""})
    for platform in ("linux", "darwin"):
        assert marker.evaluate({"sys_platform": platform, "extra": ""}), platform
    (own,) = (Requirement(r) for r in deps["llama"] if r.startswith("llama-cpp-python"))
    assert llama[0].specifier == own.specifier, (llama[0], own)
