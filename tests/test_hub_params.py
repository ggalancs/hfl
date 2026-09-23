# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Total vs active parameters, and the estimates that depend on them.

The bug these pin: the size heuristic took the last ``<N>B`` in a repo
name, so ``Qwen3-30B-A3B`` — where ``A3B`` is the *active* count — was
read as a 3B model and its VRAM estimated 6.5x under. The smart-pull
budget check runs on that number, so a 24 GB machine happily planned a
235B pull and only discovered the mistake after ~130 GB of download.

Two halves are tested, and both are needed:

* the parser must split total from active, and
* it must return ``None`` rather than a plausible number when it cannot
  tell — a default is the thing that caused the harm in the first place.

Totals below were measured from each repo's safetensors metadata on the
Hub on 2026-09-23 (``HfApi().model_info(repo, expand=["safetensors"])``);
active counts are the vendor's published figure for the original rows and,
for the rows added that day, the figure in the repo name. They are constants, not fixtures:
the tests must not reach the network, both because tests should be
hermetic and because the project's whole premise is not depending on
somebody's service to work.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hfl.hub.params import (
    ParamEstimate,
    estimate_params,
    parse_params_from_name,
    total_b_from_hub_files,
)
from hfl.hub.quant_table import estimate_vram_gb

# (repo_id, measured total B, active B)
KNOWN_MOE = [
    ("Qwen/Qwen3-30B-A3B", 30.53, 3.3),
    ("Qwen/Qwen3-235B-A22B", 235.09, 22.0),
    ("Qwen/Qwen3-Next-80B-A3B-Instruct", 81.32, 3.0),
    ("Qwen/Qwen3-Coder-480B-A35B-Instruct", 480.15, 35.0),
    ("Qwen/Qwen2-57B-A14B", 57.41, 14.0),
    ("baidu/ERNIE-4.5-21B-A3B-PT", 21.95, 3.0),
    ("LiquidAI/LFM2-8B-A1B", 8.34, 1.0),
    # Active count in millions.
    ("ibm-granite/granite-3.0-3b-a800m-instruct", 3.37, 0.8),
    # Total in trillions; the dense rule used to read this as a 95B model.
    ("Qwen/Qwen3.8-2.4T-A95B", 2446.18, 95.0),
    ("unsloth/Qwen3-30B-A3B-GGUF", 30.53, 3.3),
    ("bartowski/Qwen3-235B-A22B-GGUF", 235.09, 22.0),
]

# Sub-2B models are excluded on purpose: see
# ``test_vendor_rounding_is_a_known_limit`` — a vendor's "1B" is 1.24B, a
# 19% gap that no name parser can close, and pretending otherwise by
# widening the tolerance would hide real drift on the large models.
KNOWN_DENSE = [
    ("meta-llama/Llama-3.1-70B-Instruct", 70.6),
    ("Qwen/Qwen2.5-7B-Instruct", 7.6),
    ("Qwen/Qwen2.5-0.5B", 0.49),
    ("google/gemma-3-27b-it", 27.4),
    ("TheBloke/Llama-2-13B-GGUF", 13.0),
    ("mistralai/Mistral-7B-v0.3", 7.25),
]

# Names that prove the model is MoE without revealing its total. Turning
# ``8x7B`` into 56B would overshoot (attention and embeddings are shared,
# the real figure is 46.7B) and the correction factor that fixes Mixtral
# is fitted to one family — so the total must stay unknown.
#
# The same holds for names that carry only the active count. Read as dense,
# each of these came out 5-24x under its real total — the direction that
# plans a download the machine cannot hold.
MOE_WITHOUT_TOTAL = [
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", 7.0),
    ("mistralai/Mixtral-8x22B-v0.1", 22.0),
    # <active>B-<experts>E: read as 17B, really 108.6B and 401.6B.
    ("meta-llama/Llama-4-Scout-17B-16E-Instruct", 17.0),
    ("meta-llama/Llama-4-Maverick-17B-128E-Instruct", 17.0),
    # A<active>B with no total: read as 2.7B and 13B, really 14.3B and 80.4B.
    ("Qwen/Qwen1.5-MoE-A2.7B", 2.7),
    ("tencent/Hunyuan-A13B-Instruct", 13.0),
]

NO_SIGNAL = [
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V2-Lite",
    "moonshotai/Kimi-K2-Instruct",
    "microsoft/Phi-3.5-MoE-instruct",
    "zai-org/GLM-4.5-Air",
    "ai21labs/AI21-Jamba-Mini-1.5",
    "microsoft/phi-2",
]

# Every MoE repo above plus those whose name reads as dense, with the total
# measured on the Hub. See TestNeverUnderEstimates.
MEASURED_MOE_TOTALS = {
    **{repo: total for repo, total, _ in KNOWN_MOE},
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 46.7,
    "mistralai/Mixtral-8x22B-v0.1": 140.62,
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": 108.64,
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 401.58,
    "Qwen/Qwen1.5-MoE-A2.7B": 14.32,
    "tencent/Hunyuan-A13B-Instruct": 80.39,
    "deepseek-ai/DeepSeek-V3": 684.53,
    "deepseek-ai/DeepSeek-V2-Lite": 15.71,
    "moonshotai/Kimi-K2-Instruct": 1026.41,
    "microsoft/Phi-3.5-MoE-instruct": 41.87,
    "zai-org/GLM-4.5-Air": 110.47,
    "ai21labs/AI21-Jamba-Mini-1.5": 51.57,
    # MoE, but named by total alone — so the name reads as dense and the
    # total is still right. Active is over-estimated, the safe direction.
    "deepseek-ai/deepseek-moe-16b-base": 16.38,
    "allenai/OLMoE-1B-7B-0924": 6.92,
    "openai/gpt-oss-20b": 20.91,
    "openai/gpt-oss-120b": 116.83,
    "jetmoe/jetmoe-8b": 8.52,
}

TOLERANCE = 0.15


class TestMoENames:
    @pytest.mark.parametrize(("repo", "total", "active"), KNOWN_MOE)
    def test_total_and_active_within_tolerance(self, repo, total, active):
        est = parse_params_from_name(repo)
        assert est.is_moe, f"{repo} should read as MoE"
        assert est.total_b is not None and est.active_b is not None
        assert abs(est.total_b - total) / total <= TOLERANCE, (
            f"{repo}: total {est.total_b}B is outside ±15% of the published {total}B"
        )
        assert abs(est.active_b - active) / active <= TOLERANCE

    @pytest.mark.parametrize(("repo", "total", "active"), KNOWN_MOE)
    def test_the_last_B_number_is_not_the_answer(self, repo, total, active):
        """The regression guard.

        This is the assertion that reddens if anyone restores "take the
        last ``<N>B``" — that heuristic returns the *active* count, which
        for every row here is far below the total.
        """
        est = parse_params_from_name(repo)
        assert est.total_b != pytest.approx(est.active_b), (
            f"{repo}: total and active came out equal, which means the parser "
            "fell back to the single-number heuristic the MoE split exists to replace"
        )
        assert est.total_b > est.active_b

    @pytest.mark.parametrize(("repo", "expert_b"), MOE_WITHOUT_TOTAL)
    def test_active_only_names_refuse_to_guess_a_total(self, repo, expert_b):
        est = parse_params_from_name(repo)
        assert est.is_moe
        assert est.total_b is None, (
            f"{repo}: the total was guessed from N x S, which overshoots — "
            "the Hub must be asked instead"
        )
        assert est.active_b == pytest.approx(expert_b)
        assert not est.known

    @pytest.mark.parametrize("repo", NO_SIGNAL)
    def test_unreadable_names_return_nothing(self, repo):
        est = parse_params_from_name(repo)
        assert est.total_b is None
        assert est.source == "unknown"
        assert not est.known


class TestNeverUnderEstimates:
    """The invariant the smart-pull budget depends on, over every real MoE
    repo measured: if the name yields a total, it is not below the real
    one. Unknown is allowed — it sends the planner to the Hub's file sizes.
    Too low is not: it plans a pull the machine cannot hold."""

    @pytest.mark.parametrize(("repo", "real_total"), sorted(MEASURED_MOE_TOTALS.items()))
    def test_name_total_is_unknown_or_not_below_the_real_one(self, repo, real_total):
        est = parse_params_from_name(repo)
        if est.total_b is None:
            return
        assert est.total_b >= real_total * (1 - TOLERANCE), (
            f"{repo}: the name reads as {est.total_b}B but the Hub measures "
            f"{real_total}B — an under-estimate that would pass the budget check"
        )

    def test_the_table_is_wide(self):
        """The first version pinned three families. Naming conventions keep
        arriving (A-prefixed actives, expert counts, trillions); the table
        should keep covering them."""
        assert len(MEASURED_MOE_TOTALS) >= 25


class TestDenseUnchanged:
    """The MoE work must not move a single dense answer."""

    @pytest.mark.parametrize(("repo", "total"), KNOWN_DENSE)
    def test_dense_total_within_tolerance(self, repo, total):
        est = parse_params_from_name(repo)
        assert not est.is_moe
        assert est.total_b is not None
        assert abs(est.total_b - total) / total <= TOLERANCE

    @pytest.mark.parametrize(("repo", "total"), KNOWN_DENSE)
    def test_dense_total_equals_active(self, repo, total):
        est = parse_params_from_name(repo)
        assert est.total_b == est.active_b

    def test_vendor_rounding_is_a_known_limit(self):
        """A name carries the vendor's nominal figure, not the real count.

        ``Llama-3.2-1B`` really has 1.24B parameters. Reading names can
        never close that 19% gap, and it matters most where the absolute
        number is smallest — which is where an over-estimate is cheap.
        Written down as a test so it stays a known limit rather than a
        surprise, and so nobody "fixes" the parser to chase it.
        """
        est = parse_params_from_name("meta-llama/Llama-3.2-1B")
        assert est.total_b == 1.0
        published = 1.24
        assert abs(est.total_b - published) / published > TOLERANCE


class TestHubFileFallback:
    """When the name cannot answer, bytes on disk can."""

    @staticmethod
    def _api_with(files):
        api = MagicMock()
        info = MagicMock()
        info.siblings = [MagicMock(rfilename=n, size=sz) for n, sz in files]
        api.model_info = MagicMock(return_value=info)
        return api

    def test_sums_weight_files_only(self):
        api = self._api_with(
            [
                ("model-00001-of-00002.safetensors", 60_000_000_000),
                ("model-00002-of-00002.safetensors", 33_400_000_000),
                ("README.md", 4_000),
                ("tokenizer.json", 17_000_000),
            ]
        )
        # bf16 safetensors -> 2 bytes per parameter -> ~46.7B, Mixtral-8x7B.
        assert total_b_from_hub_files(api, "mistralai/Mixtral-8x7B") == pytest.approx(
            46.7, rel=0.02
        )

    def test_quantised_filename_changes_bytes_per_param(self):
        api = self._api_with([("model-Q4_K_M.gguf", 17_000_000_000)])
        got = total_b_from_hub_files(api, "x/y")
        assert got is not None and 28 < got < 33  # ~30B at Q4_K_M

    def test_no_sizes_reported_is_None_not_zero(self):
        """ "Could not measure" and "measured as tiny" must never collapse."""
        api = self._api_with([("model.safetensors", None)])
        assert total_b_from_hub_files(api, "x/y") is None

    def test_hub_failure_is_None_not_an_exception(self):
        api = MagicMock()
        api.model_info = MagicMock(side_effect=RuntimeError("offline"))
        assert total_b_from_hub_files(api, "x/y") is None

    def test_estimate_params_falls_through_to_the_hub(self):
        api = self._api_with([("model.safetensors", 93_400_000_000)])
        est = estimate_params("mistralai/Mixtral-8x7B", api=api)
        assert est.source == "hub-files"
        assert est.is_moe, "the name already proved it is MoE; that must survive"
        assert est.total_b == pytest.approx(46.7, rel=0.02)
        assert est.active_b == pytest.approx(7.0), "the expert size from the name"

    def test_estimate_params_never_touches_the_hub_when_the_name_answers(self):
        api = MagicMock()
        est = estimate_params("meta-llama/Llama-3.1-70B-Instruct", api=api)
        assert est.total_b == 70.0
        api.model_info.assert_not_called()

    def test_no_api_means_no_network(self):
        est = estimate_params("deepseek-ai/DeepSeek-V3")
        assert est.total_b is None


class TestVramSplit:
    def test_weights_follow_total_and_kv_follows_active(self):
        """A 30B-A3B carries a 30B model's weights and a small model's cache."""
        moe = estimate_vram_gb(params_b=30.0, quantization="q4_k_m", active_params_b=3.0)
        dense30 = estimate_vram_gb(params_b=30.0, quantization="q4_k_m")

        assert moe.weights_gb == pytest.approx(dense30.weights_gb), (
            "every expert is resident, so the weights must match the dense 30B"
        )
        assert moe.kv_cache_gb < dense30.kv_cache_gb, (
            "the architecture is that of a small model, so its cache is smaller"
        )

    def test_dense_callers_are_bit_for_bit_unchanged(self):
        for params in (1.0, 7.0, 13.0, 70.0):
            with_default = estimate_vram_gb(params_b=params, quantization="q4_k_m")
            explicit = estimate_vram_gb(
                params_b=params, quantization="q4_k_m", active_params_b=params
            )
            assert with_default == explicit

    def test_the_old_estimate_was_the_dangerous_direction(self):
        """What the bug produced, stated as a number so it cannot come back."""
        broken = estimate_vram_gb(params_b=22.0, quantization="q4_k_m")  # read "A22B"
        correct = estimate_vram_gb(params_b=235.0, quantization="q4_k_m", active_params_b=22.0)
        assert correct.total_gb > broken.total_gb * 5


class TestSmartPullRefusesToGuess:
    def test_unknown_size_returns_a_reason_not_a_7B_assumption(self):
        from hfl.hub import smart_pull as sp

        api = MagicMock()
        api.model_info = MagicMock(side_effect=RuntimeError("offline"))

        plan, reason = sp.try_smart_plan("deepseek-ai/DeepSeek-V3", api=api, max_vram_gb=24.0)

        assert plan is None
        assert reason is not None
        assert "cannot determine the size" in reason
        # The old behaviour: silently treat it as 7B and plan a pull that
        # cannot possibly fit. The reason must instead tell the operator
        # what to do next.
        assert "hfl pull" in reason


class TestParamEstimateShape:
    def test_known_is_about_the_total_only(self):
        assert not ParamEstimate(None, 7.0, True, "name").known
        assert ParamEstimate(46.7, 7.0, True, "hub-files").known

    def test_active_above_total_is_not_moe(self):
        """A name like ``7B-A70B`` is nonsense; treat it as dense, not as a
        model whose experts outweigh the model."""
        est = parse_params_from_name("org/weird-7B-A70B")
        assert not est.is_moe
        assert est.total_b == est.active_b == 7.0
