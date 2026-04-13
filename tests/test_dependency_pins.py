# SPDX-License-Identifier: HRUL-1.0
"""Regression guards for cross-package dependency compatibility.

This file exists because of a real incident: a dev pre-release of
``transformers`` (5.3.0.dev0) was installed in a host Python that also
had ``huggingface_hub==0.27.1``. ``transformers`` tried to import the
``is_offline_mode`` symbol from ``huggingface_hub`` and crashed with
``ImportError``. HFL's ``gguf_converter`` invokes
``convert_hf_to_gguf.py`` via ``sys.executable``, so a broken host
Python silently breaks model conversion at runtime — and the test
suite did not catch it because the conflicting packages aren't in the
``[dev]`` extra that CI installs.

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

    def test_huggingface_hub_pin_excludes_1x(self):
        cfg = _load_pyproject()
        deps = cfg["project"]["dependencies"]
        pin = _find_pin(deps, "huggingface-hub")
        # transformers 5.x requires huggingface_hub>=1.5.0 with a
        # different API. We must stay on the 0.x line to keep the
        # rest of HFL's surface working.
        assert "<1.0" in pin or "<1," in pin, (
            f"huggingface-hub pin must exclude 1.x: {pin!r}\n"
            "If you need to allow 1.x, audit every transformers import "
            "and bump the [transformers] / [tts] pins together."
        )

    @pytest.mark.parametrize("extra_name", ["transformers", "tts"])
    def test_transformers_extras_exclude_5x(self, extra_name):
        cfg = _load_pyproject()
        extras = cfg["project"]["optional-dependencies"][extra_name]
        pin = _find_pin(extras, "transformers")
        # transformers 5.x requires huggingface_hub>=1.5.0 (different
        # API surface). Until HFL is rewritten to target that surface
        # we must keep the [transformers] / [tts] pins on 4.x.
        assert "<5.0" in pin or "<5," in pin, (
            f"[{extra_name}] transformers pin must exclude 5.x: {pin!r}\n"
            "Loosening this pin re-introduces the is_offline_mode "
            "ImportError that crashed convert_hf_to_gguf.py."
        )

    def test_transformers_extras_have_minimum(self):
        """Sanity: a lower bound is also required so users don't end
        up on an ancient transformers that lacks features we use."""
        cfg = _load_pyproject()
        for extra_name in ("transformers", "tts"):
            extras = cfg["project"]["optional-dependencies"][extra_name]
            pin = _find_pin(extras, "transformers")
            assert ">=4." in pin, f"[{extra_name}] transformers pin needs a >=4.x lower bound"

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
        major version axis. Specifically: transformers 4.x must coexist
        with huggingface_hub 0.x. transformers 5.x ↔ huggingface_hub
        1.x is also acceptable in principle but HFL doesn't support it
        yet."""
        import huggingface_hub
        import transformers

        tx_major = int(transformers.__version__.split(".")[0])
        hh_major = int(huggingface_hub.__version__.split(".")[0])
        if tx_major == 4:
            assert hh_major == 0, (
                f"transformers 4.x requires huggingface_hub 0.x, got "
                f"{huggingface_hub.__version__}. Run "
                f"``pip install --upgrade --force-reinstall "
                f"'huggingface-hub>=0.27.0,<1.0'``."
            )
        elif tx_major == 5:
            pytest.fail(
                f"transformers 5.x ({transformers.__version__}) is not "
                "supported by HFL — its huggingface_hub contract is "
                "incompatible with our other pins. Downgrade with "
                "``pip install --upgrade --force-reinstall "
                "'transformers>=4.47.0,<5.0'``."
            )
        else:
            pytest.fail(f"unexpected transformers major: {tx_major}")
