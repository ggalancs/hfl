# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Parameter counts for a Hub repo — total vs active.

A dense model has one number and it answers both questions a serving
decision asks. A Mixture-of-Experts model has two, and confusing them is
expensive in both directions:

* **Total** parameters decide how much memory the weights occupy. Every
  expert is resident even though only a few run per token.
* **Active** parameters track the architecture — layer count and hidden
  size — and therefore the KV cache and the speed.

The previous heuristic took the last ``<N>B`` in the repo name, which is
correct for ``Llama-3.1-8B-Instruct`` and wrong for the family that now
dominates the Hub: in ``Qwen3-30B-A3B`` the trailing ``A3B`` is the
*active* count, so the estimate came out 6.5x under. Measured against the
shipped estimator at Q4_K_M, before this module existed::

    Qwen3-30B-A3B      4.3 GB estimated    27.9 GB actual    6.5x
    Qwen3-235B-A22B   20.4 GB estimated   182.0 GB actual    8.9x
    Mixtral-8x7B       8.3 GB estimated    35.8 GB actual    4.3x
    DeepSeek-V3        8.3 GB estimated   477.4 GB actual   57.2x

The last row is the worst case and the most instructive: the name carries
no ``B`` at all, so it fell through to a hard-coded 7B default. A default
is a guess wearing a number's clothes. When the name cannot answer, this
module asks the Hub for the actual file sizes instead — bytes on disk are
not an estimate.

``N x S`` names (``Mixtral-8x7B``) are deliberately *not* turned into
``N * S``: the experts share attention and embeddings, so the product
overshoots, and the correction factor that fixes Mixtral is fitted to one
model family. Such a name marks the repo as MoE with an unknown total and
defers to the Hub, which is the honest answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

__all__ = ["ParamEstimate", "estimate_params", "parse_params_from_name"]


# Bytes per parameter for the weight formats a Hub repo ships, used to turn
# summed file sizes back into a parameter count. Keyed by the marker that
# appears in the filename; the value is the dominant element width.
_BYTES_PER_PARAM_BY_MARKER: tuple[tuple[str, float], ...] = (
    ("q2", 0.34),
    ("q3", 0.44),
    ("q4", 0.56),
    ("iq4", 0.53),
    ("q5", 0.68),
    ("q6", 0.82),
    ("q8", 1.06),
    ("f32", 4.0),
    ("bf16", 2.0),
    ("f16", 2.0),
    ("4bit", 0.56),
    ("8bit", 1.06),
)

_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".npz")

# ``30B-A3B``, ``30B A3B``, ``80b-a3b`` — total first, active after the ``A``.
_MOE_EXPLICIT = re.compile(
    r"(?<!\d)(\d{1,6}(?:\.\d{1,3})?)\s*[Bb]\b[\s._-]*[Aa](\d{1,6}(?:\.\d{1,3})?)\s*[Bb]\b"
)

# ``8x7B``, ``8 x 22B`` — experts times expert size. Signals MoE; the total
# is NOT the product, so it stays unknown. See the module docstring.
_MOE_PRODUCT = re.compile(r"(?<!\d)(\d{1,3})\s*[xX]\s*(\d{1,6}(?:\.\d{1,3})?)\s*[Bb]\b")

# Plain ``<N>B``. Bounded quantifiers keep this linear on hostile input
# (CodeQL py/polynomial-redos); the bounds sit far above any real count.
_DENSE = re.compile(r"(?<!\d)(\d{1,6}(?:\.\d{1,3})?)\s*[Bb](?![A-Za-z])")


@dataclass(frozen=True)
class ParamEstimate:
    """What is known about a repo's size, and how it was learnt.

    ``total_b`` may be ``None`` while ``is_moe`` is ``True``: that is the
    ``8x7B`` case, where the name proves the model is MoE without
    revealing how large it is. Callers must treat ``None`` as "ask the
    Hub or refuse", never as a licence to substitute a default.
    """

    total_b: float | None
    """Billions of parameters resident in memory. Sizes the weights."""

    active_b: float | None
    """Billions of parameters that run per token. Tracks the architecture,
    so it sizes the KV cache. Equals ``total_b`` for a dense model."""

    is_moe: bool
    """Whether the repo is a Mixture-of-Experts model."""

    source: str
    """``"name"``, ``"hub-files"`` or ``"unknown"`` — so a caller can say
    how confident the number is instead of presenting all three alike."""

    @property
    def known(self) -> bool:
        """Whether a memory decision can be made from this at all."""
        return self.total_b is not None


def parse_params_from_name(text: str) -> ParamEstimate:
    """Read parameter counts off a repo id / tag string.

    Order matters: the MoE patterns are tried first, because the dense
    pattern would happily match the ``A3B`` half of ``30B-A3B`` and
    return the active count as if it were the whole model.
    """
    moe = _MOE_EXPLICIT.search(text)
    if moe:
        try:
            total = float(moe.group(1))
            active = float(moe.group(2))
        except ValueError:  # pragma: no cover — the pattern only matches digits
            return ParamEstimate(None, None, False, "unknown")
        # Guard against a name that reads as MoE but is not: the active
        # half cannot exceed the total.
        if active > total:
            return ParamEstimate(total, total, False, "name")
        return ParamEstimate(total, active, True, "name")

    product = _MOE_PRODUCT.search(text)
    if product:
        try:
            expert_b = float(product.group(2))
        except ValueError:  # pragma: no cover
            expert_b = 0.0
        # The product overshoots (shared attention / embeddings) and the
        # correction is model-family-specific, so the total stays unknown
        # and the Hub gets asked. The expert size is a usable *active*
        # figure: one expert's worth of parameters runs per token.
        return ParamEstimate(None, expert_b or None, True, "name")

    dense = _DENSE.findall(text)
    if dense:
        try:
            # Last match: repo names put the size right before the variant
            # tag (``Llama-3.1-8B-Instruct``).
            value = float(dense[-1])
        except ValueError:  # pragma: no cover
            return ParamEstimate(None, None, False, "unknown")
        return ParamEstimate(value, value, False, "name")

    return ParamEstimate(None, None, False, "unknown")


def _bytes_per_param(filename: str) -> float:
    """Guess the element width from a weight filename.

    Checked longest-marker-first so ``iq4`` is not shadowed by ``q4`` and
    ``bf16`` not by ``f16``.
    """
    low = filename.lower()
    best: tuple[int, float] | None = None
    for marker, width in _BYTES_PER_PARAM_BY_MARKER:
        if marker in low and (best is None or len(marker) > best[0]):
            best = (len(marker), width)
    if best is not None:
        return best[1]
    # Unmarked safetensors from the Hub are bf16 far more often than not.
    return 2.0


def total_b_from_hub_files(api: "HfApi", repo_id: str) -> float | None:
    """Derive a parameter count from the repo's actual weight file sizes.

    Bytes on disk are a measurement, not a heuristic: whatever the naming
    convention, an MoE's experts are in those files. Returns ``None`` when
    the Hub reports no sizes, so the caller can distinguish "measured as
    small" from "could not measure" — the two must never collapse.
    """
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except Exception:
        logger.debug("could not read file metadata for %s", repo_id, exc_info=True)
        return None

    siblings = getattr(info, "siblings", None) or []
    total_bytes = 0.0
    per_param = 2.0
    for sib in siblings:
        name = getattr(sib, "rfilename", "") or ""
        if not name.lower().endswith(_WEIGHT_SUFFIXES):
            continue
        size = getattr(sib, "size", None)
        if not size:
            continue
        total_bytes += float(size)
        per_param = _bytes_per_param(name)

    if total_bytes <= 0:
        return None
    return round(total_bytes / per_param / 1e9, 2)


def estimate_params(
    repo_id: str,
    tags: Iterable[str] = (),
    *,
    api: "HfApi | None" = None,
) -> ParamEstimate:
    """Best available reading of a repo's total and active parameters.

    The name is tried first because it costs nothing. The Hub is asked
    only when the name leaves the total unknown — the ``8x7B`` and
    ``DeepSeek-V3`` cases — and only when the caller supplied an ``api``,
    so this stays a pure function for anyone who does not want network.
    """
    text = repo_id + " " + " ".join(tags)
    est = parse_params_from_name(text)
    if est.known or api is None:
        return est

    measured = total_b_from_hub_files(api, repo_id)
    if measured is None:
        return est
    return ParamEstimate(
        total_b=measured,
        # A name that said "MoE" without a total still told us the expert
        # size; keep it. Otherwise the model is dense and the two agree.
        active_b=est.active_b if est.is_moe else measured,
        is_moe=est.is_moe,
        source="hub-files",
    )
