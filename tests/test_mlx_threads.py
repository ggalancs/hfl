# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""An MLX model loaded on one thread and used on another.

Since mlx 0.32 each thread has its own streams, and a lazy array stays tied
to the stream of the thread that built it. mlx-lm's Llama 3, YaRN, LongRoPE
and proportional (Gemma 4) RoPEs keep a lazy ``_freqs`` that loading does not
evaluate; HFL loads on a worker thread and generates on the batch thread, so
every Gemma 4 request failed: "There is no Stream(gpu, 1) in current thread"
(mlx 0.32.0-0.32.3, mlx-lm 0.31.3-0.32.0). Real mlx, no weights: the RoPE
modules alone carry the problem.
"""

from __future__ import annotations

import importlib.util
import threading
from typing import Any, Callable

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None or importlib.util.find_spec("mlx_lm") is None,
    reason="needs mlx and mlx-lm (Apple Silicon)",
)

ROPES = {
    "llama3": {
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    },
    "yarn": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096},
    "longrope": {
        "rope_type": "longrope",
        "original_max_position_embeddings": 4096,
        "short_factor": [1.0] * 32,
        "long_factor": [1.0] * 32,
    },
    "proportional": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "factor": 1.0},
}


def _on_another_thread(fn: Callable[[], Any]) -> Any:
    """``fn()`` on a fresh thread: its result, or the exception it raised."""
    out: dict[str, Any] = {}

    def run() -> None:
        try:
            out["value"] = fn()
        except Exception as exc:  # returned, so the test can show it
            out["value"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    return out["value"]


@pytest.mark.parametrize("config", list(ROPES.values()), ids=list(ROPES))
def test_a_model_loaded_on_one_thread_runs_on_another(config: dict) -> None:
    import mlx.core as mx
    from mlx_lm.models.rope_utils import initialize_rope

    from hfl.engine.mlx_engine import _settle_model_arrays

    def load() -> Any:
        rope = initialize_rope(64, 10000.0, False, config, 131072)
        _settle_model_arrays(rope, None)
        return rope

    rope = _on_another_thread(load)
    x = mx.zeros((1, 1, 4, 64))
    mx.eval(x)
    used = _on_another_thread(lambda: mx.eval(rope(x)) or "ran")
    assert used == "ran", f"{type(rope).__name__} on another thread: {used!r}"
