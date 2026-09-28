# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Time to first token, generation speed and what is loaded, in /metrics.

Speed is tokens over the decode phase only: prompt processing runs ~10x
faster, and tokens ÷ total time mixes the two into a number that is neither.
Memory figures are the owner's (a local caller), as in /api/ps."""

from __future__ import annotations

from hfl.metrics import Metrics


def _value(text: str, line_start: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(line_start)).split()[-1]


def test_speed_is_over_the_decode_phase_only() -> None:
    m = Metrics()
    # 101 tokens in 3 s, of which the first took 1 s (prompt): 100 tok / 2 s.
    m.record_generation(3000, 20, 101, first_token_ms=1000, decode_ms=2000, decode_tokens=100)
    out = m.export_prometheus()
    assert _value(out, "hfl_generation_tokens_per_second_sum") == "50.00"
    assert 'hfl_generation_tokens_per_second_bucket{le="30.0"} 0' in out
    assert 'hfl_generation_tokens_per_second_bucket{le="50.0"} 1' in out
    assert _value(out, "hfl_time_to_first_token_ms_sum") == "1000.00"
    assert 'hfl_time_to_first_token_ms_bucket{le="1000.0"} 1' in out


def test_a_generation_without_timing_adds_no_speed() -> None:
    m = Metrics()
    m.record_generation(500, 10, 5)
    out = m.export_prometheus()
    assert "hfl_generation_tokens_per_second" not in out
    assert "hfl_time_to_first_token_ms" not in out


def test_memory_only_for_the_owner(monkeypatch) -> None:
    from hfl.engine import residency

    class _Mem:
        total, in_use = 64 * 2**30, 20 * 2**30

    monkeypatch.setattr(residency, "current_memory", lambda: _Mem())
    monkeypatch.setattr(residency, "current_gpu_memory", lambda: None)
    m = Metrics()
    remote = m.export_prometheus(include_host=False)
    local = m.export_prometheus(include_host=True)
    assert "hfl_models_loaded 0" in remote and "hfl_models_loaded 0" in local
    assert "hfl_memory_total_bytes" not in remote
    assert _value(local, "hfl_memory_total_bytes") == str(64 * 2**30)
    assert _value(local, "hfl_memory_in_use_bytes") == str(20 * 2**30)
    assert "hfl_memory_budget_bytes" in local
