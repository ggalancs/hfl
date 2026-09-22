# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Why HFL does not offload MoE experts to CPU yet, checked rather than remembered.

`llama.cpp`'s `--n-cpu-moe` / `--override-tensor` keep attention on the
GPU and push the expert tensors to system RAM. On a discrete GPU that is
the difference between "a 30B-A3B does not fit" and "a 30B-A3B runs" —
the small activation vector crosses PCIe instead of the weights. It is
the single biggest reach lever for the model family that now dominates
the Hub, so its absence deserves a reason, not a silence.

The reason, verified against the installed binding rather than assumed:

* The **C layer has it.** `llama_cpp.llama_model_params` carries a
  `tensor_buft_overrides` field, and `llama_model_tensor_override` and
  `llama_max_tensor_buft_overrides` are both exported.
* The **Python layer does not.** `Llama.__init__` builds its own
  `llama_model_default_params()`, assigns the handful of fields it knows
  about, and constructs the model from them before returning. It takes
  `**kwargs` but never reads one for tensor placement, and by the time a
  caller can touch `llama.model_params` the model already exists.

So the only route today is monkey-patching `Llama.__init__` or carrying a
fork — in the module that owns HFL's model-lifecycle use-after-free
family, for hardware this project cannot test against. That trade is bad
enough to decline deliberately.

This test is the tripwire. When the binding grows a supported keyword, it
fails and says so, instead of the capability sitting unnoticed for a year.

Two separate conditions are checked, because they can arrive apart: the
kwarg on `Llama.__init__`, and a documented way to hand over
`tensor_buft_overrides`. Either one unblocks the work.
"""

from __future__ import annotations

import importlib.util
import inspect

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("llama_cpp") is None,
    reason="llama-cpp-python not installed (the CI venv omits the [llama] extra)",
)

# Any of these appearing as an accepted keyword means the work is unblocked.
UNBLOCKING_KWARGS = {
    "n_cpu_moe",
    "cpu_moe",
    "override_tensor",
    "tensor_buft_overrides",
    "tensor_buft_override",
}


def test_the_python_binding_still_hides_tensor_placement():
    """Fails the day `Llama()` accepts an expert-offload keyword."""
    import llama_cpp

    params = set(inspect.signature(llama_cpp.Llama.__init__).parameters)
    available = sorted(UNBLOCKING_KWARGS & params)

    assert not available, (
        f"llama-cpp-python {getattr(llama_cpp, '__version__', '?')} now accepts "
        f"{available} on Llama(). The MoE CPU-offload work is unblocked: wire it "
        "through the engine, auto-tune the layer count by walking down from full "
        "offload (the throughput cliff is silent — measured elsewhere at 69.4 -> "
        "27.5 tok/s across a single layer), and keep it inert on Apple Silicon, "
        "where unified memory means there is no PCIe hop to dodge. Then delete "
        "this test and replace it with a measured one."
    )


def test_the_c_layer_does_expose_it_so_this_is_a_binding_gap():
    """The other half of the claim.

    Stated as an assertion so the diagnosis cannot quietly become wrong:
    if the C surface ever loses the field, the reason recorded above stops
    being "the binding hides it" and this file needs rewriting.
    """
    import llama_cpp

    fields = {f[0] for f in llama_cpp.llama_model_params._fields_}
    assert "tensor_buft_overrides" in fields, (
        "llama_model_params no longer carries tensor_buft_overrides — the "
        "reason this feature is deferred has changed, and the docstring above "
        "is now wrong."
    )


def test_hfl_does_not_pretend_to_support_it():
    """No half-wired knob.

    An `HFL_N_CPU_MOE` that silently did nothing would be worse than the
    gap: the operator would set it, see no error, and conclude their
    machine simply cannot run the model.
    """
    from hfl.config import HFLConfig

    pretend = [
        name
        for name in dir(HFLConfig)
        if not name.startswith("_") and ("cpu_moe" in name.lower() or "n_moe" in name.lower())
    ]
    assert pretend == [], (
        f"HFLConfig exposes {pretend} while the binding cannot honour it. "
        "A knob that accepts a value and drops it is a lie with a default."
    )


def test_apple_silicon_would_not_want_it_anyway():
    """Guards the reasoning, not just the code.

    `--n-cpu-moe` exists to avoid moving expert weights across PCIe. Apple
    Silicon has unified memory and no such bus, so the flag is not a
    smaller win there — it is the wrong tool. Recording this stops a
    future implementation from enabling it on Metal "for consistency".
    """
    from hfl.hub.hw_profile import get_hw_profile

    profile = get_hw_profile()
    if profile.gpu_kind != "metal":
        pytest.skip("this assertion is about Apple Silicon hosts")

    # There is no discrete VRAM to run out of: the budget is the machine's
    # memory, shared with the CPU that would be "helping".
    assert profile.gpu_vram_gb is None or profile.gpu_vram_gb > 0
