# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Short model names: ``hfl run qwen3-coder`` without a Hub reference.

A name that is neither a local model nor ``org/repo`` is looked up on the
Hub: GGUF repos whose name contains it (and a size tag such as ``:8b``, if
given), ranked by downloads, with instruct builds and the well-known
quantizers favoured and derivatives (abliterated, uncensored, merges...)
left out. The quantization is chosen from the repo's real file sizes: the
one asked for, else Q4_K_M, else the next one down that fits this machine.

The choice is only a proposal: the CLI shows it and asks before a
download, and remembers it as an alias, so the name is local from then on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Asked for when nothing else is: the usual balance of size and quality.
DEFAULT_QUANT = "Q4_K_M"
# When the default does not fit, the next ones down, best first.
_SMALLER = ("Q4_K_S", "IQ4_XS", "Q3_K_M", "IQ3_M", "Q3_K_S", "Q2_K")
# Larger quants a user may ask for explicitly, e.g. ``qwen3:8b-q8_0``.
_KNOWN = ("Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", DEFAULT_QUANT, *_SMALLER, "F16", "BF16")

# Publishers whose GGUF builds are the usual reference ones.
_TRUSTED = {
    "ggml-org",
    "unsloth",
    "bartowski",
    "lmstudio-community",
    "qwen",
    "google",
    "meta-llama",
    "mistralai",
    "microsoft",
    "ibm-granite",
    "nousresearch",
    "deepseek-ai",
}
# Derivatives nobody asking for "llama3.2" means.
_DERIVATIVE = re.compile(
    r"abliterat|obliterat|uncensor|derestrict|heretic|nsfw|roleplay|\brp\b|"
    r"merge|franken|lora|orpo|dpo|imatrix-calib|draft|pruned|\breap\b|"
    r"reasoning-distill",
    re.IGNORECASE,
)
# Other tasks sharing a family's name; only when the name asks for them.
_OTHER_TASK = ("asr", "rerank", "embed", "tts", "guard", "speech", "audio", "ocr")
_QUANT_IN_FILE = re.compile(
    r"(?<![A-Za-z0-9])(IQ\d_[A-Z]+|Q\d_K_[SML]|Q\d_K|Q\d_\d|BF16|F16)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_SIZE_TAG = re.compile(r"^\d+(\.\d+)?[bm]$", re.IGNORECASE)
_HEADROOM = 1.15  # weights + a working context


@dataclass
class ShortNameMatch:
    """The model a short name resolved to — a proposal, not a download."""

    name: str
    repo_id: str
    quantization: str
    size_bytes: int
    downloads: int
    others: list[str] = field(default_factory=list)

    @property
    def reference(self) -> str:
        return f"hf.co/{self.repo_id}:{self.quantization}"


_SHORT_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*(:[a-z0-9._-]+)?$", re.IGNORECASE)


def is_short_name(name: str) -> bool:
    """Whether ``name`` reads like a model name to look up (``qwen3-coder``,
    ``qwen3:8b``) — not a path, not a Hub reference, not noise."""
    name = name.strip()
    return bool(_SHORT_NAME.match(name)) and len(_norm(name.split(":", 1)[0])) >= 3


def alias_for(name: str) -> str:
    """The local alias a resolved short name is remembered under."""
    return name.strip().lower().replace(":", "-")


def split_name(name: str) -> tuple[str, str | None, str | None]:
    """``"qwen3:8b"`` → ``("qwen3", "8b", None)``; ``"qwen3:Q8_0"`` →
    ``("qwen3", None, "Q8_0")``; ``"qwen3:8b-q8_0"`` → both."""
    base, _, tag = name.strip().partition(":")
    size = quant = None
    for part in filter(None, re.split(r"[-_]", tag, maxsplit=1) if tag else []):
        if _SIZE_TAG.match(part) and size is None:
            size = part.lower()
        elif part.upper() in _KNOWN:
            quant = part.upper()
    if tag and size is None and quant is None:
        whole = tag.upper()
        if whole in _KNOWN:
            quant = whole
        elif _SIZE_TAG.match(tag):
            size = tag.lower()
    return base.lower(), size, quant


def _norm(text: str) -> str:
    return re.sub(r"[-_.\s]", "", text.lower())


def _queries(base: str, size: str | None = None) -> list[str]:
    """Search strings: as typed, and with a dash where letters meet digits
    (``llama3.2`` → ``llama-3.2``), since repos are spelled both ways. With
    a size tag, the size goes into the query too: a family with hundreds of
    GGUF repos does not show its small sizes among the most downloaded."""
    spaced = re.sub(r"(?<=[a-z])(?=\d)", "-", base)
    queries = [base, spaced]
    if size:
        queries = [f"{q}-{size}" for q in queries] + queries
    return list(dict.fromkeys(queries))


def _starts_a_word(base: str, repo_name: str) -> bool:
    """``base`` begins at a word of ``repo_name``: ``llama3.2`` in
    ``Llama-3.2-3B``, ``deepseek-r1`` in ``deepseek-ai_DeepSeek-R1`` — but
    ``phi`` not in ``Delphi``, ``x`` not in ``XYZAILab``... unless it is the
    whole start of that word, which ``x`` is: hence the 3-character minimum."""
    target = _norm(base)
    words = re.split(r"[-_.\s]", repo_name.lower())
    return any(_norm("".join(words[i:])).startswith(target) for i in range(len(words)))


def _matches(repo_id: str, base: str, size: str | None) -> bool:
    repo_name = _norm(repo_id.split("/", 1)[-1])
    if not _starts_a_word(base, repo_id.split("/", 1)[-1]):
        return False
    if size and size not in repo_id.lower().replace("_", "-").split("/", 1)[-1].split("-"):
        return False
    if any(task in repo_name and task not in _norm(base) for task in _OTHER_TASK):
        return False
    return not _DERIVATIVE.search(repo_id)


def _model_key(repo_id: str) -> str:
    """The model a repo packages, whoever published it: ``unsloth/Qwen3-8B-GGUF``
    and ``Qwen/Qwen3-8B-GGUF`` are one model to choose, not two."""
    name = repo_id.split("/", 1)[-1].lower()
    name = re.sub(r"[-_.]?gguf$", "", name)
    # A quant in the repo name (``...-Q4_K_M-GGUF``) names a file, not a model.
    name = re.sub(r"[-_.](i?q\d[_a-z0-9]*|f16|bf16)$", "", name)
    # bartowski-style ``<original-org>_<model>`` prefixes.
    head, sep, rest = name.partition("_")
    if sep and not re.search(r"\d", head):
        name = rest
    return _norm(name)


def _score(repo_id: str, downloads: int, base: str) -> float:
    org, _, name = repo_id.lower().partition("/")
    score = float(downloads or 0)
    if org in _TRUSTED:
        score *= 3
    if re.search(r"instruct|-it\b|-it-|chat", name) and "base" not in base:
        score *= 2
    if "base" in name and "base" not in base:
        score /= 4
    return score


def _quant_files(api: Any, repo_id: str) -> dict[str, int]:
    """GGUF bytes per quantization in ``repo_id`` (split files summed;
    multimodal projectors left out)."""
    info = api.model_info(repo_id, files_metadata=True)
    sizes: dict[str, int] = {}
    for sibling in getattr(info, "siblings", None) or []:
        path = getattr(sibling, "rfilename", "") or ""
        if not path.lower().endswith(".gguf") or "mmproj" in path.lower():
            continue
        found = _QUANT_IN_FILE.search(path.rsplit("/", 1)[-1])
        if found:
            quant = found.group(1).upper()
            sizes[quant] = sizes.get(quant, 0) + int(getattr(sibling, "size", 0) or 0)
    return sizes


def _pick_quant(sizes: dict[str, int], wanted: str | None, budget_bytes: int) -> str | None:
    """The quant to download: the one asked for if present (fits or not —
    the user asked), else the default, else the next ones down that fit."""
    if wanted:
        return wanted if wanted in sizes else None
    # The default, then smaller ones; a repo that only publishes larger
    # quants (Qwen's own 0.6B ships Q8_0 alone) gets the smallest that fits.
    for quant in (DEFAULT_QUANT, *_SMALLER, "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0"):
        if quant in sizes and sizes[quant] * _HEADROOM <= budget_bytes:
            return quant
    return None


def default_budget_bytes() -> int:
    """What a model may take here: the GPU's memory on CUDA, else the share
    of RAM HFL budgets for models (``HFL_MEMORY_BUDGET``)."""
    from hfl.hub.hw_profile import get_hw_profile

    profile = get_hw_profile()
    vram = getattr(profile, "gpu_vram_gb", None)
    if getattr(profile, "gpu_kind", None) == "cuda" and vram:
        return int(vram * 1024**3)
    try:
        import psutil

        from hfl.config import config

        share = float(getattr(config, "memory_budget_percent", 85) or 85) / 100
        return int(psutil.virtual_memory().total * share)
    except Exception:  # pragma: no cover - psutil is a core dependency
        return int((getattr(profile, "system_ram_gb", 8) or 8) * 0.7 * 1024**3)


def _default_api() -> Any:
    """The Hub client (a seam: the test suite never talks to the real Hub)."""
    from huggingface_hub import HfApi

    return HfApi()


def find_options(
    name: str,
    *,
    api: Any = None,
    budget_bytes: int | None = None,
    limit: int = 5,
) -> list[ShortNameMatch]:
    """GGUF builds of ``name`` that fit this machine, best first.

    Without a size tag the first is only the most likely one — a family's
    most downloaded build is often its smallest — so callers show the others
    and let the user pick.
    """
    if api is None:
        api = _default_api()
    base, size, wanted = split_name(name)
    if not base or not is_short_name(name):
        return []
    budget = budget_bytes if budget_bytes is not None else default_budget_bytes()
    seen: dict[str, int] = {}
    for query in _queries(base, size):
        for model in api.list_models(search=query, filter="gguf", sort="downloads", limit=50):
            repo_id = getattr(model, "id", "") or ""
            if repo_id and repo_id not in seen and _matches(repo_id, base, size):
                seen[repo_id] = int(getattr(model, "downloads", 0) or 0)
    ranked = sorted(seen, key=lambda r: _score(r, seen[r], base), reverse=True)
    options: list[ShortNameMatch] = []
    chosen: set[str] = set()
    for repo_id in ranked[: limit * 4]:
        key = _model_key(repo_id)
        if key in chosen:  # this model already has its best usable build
            continue
        sizes = _quant_files(api, repo_id)
        quant = _pick_quant(sizes, wanted, budget)
        if quant is not None:
            chosen.add(key)
            options.append(
                ShortNameMatch(
                    name=name,
                    repo_id=repo_id,
                    quantization=quant,
                    size_bytes=sizes[quant],
                    downloads=seen[repo_id],
                )
            )
        if len(options) == limit:
            break
    for option in options:
        option.others = [o.repo_id for o in options if o is not option]
    return options


def find(name: str, **kwargs: Any) -> ShortNameMatch | None:
    """The most likely GGUF build of ``name`` for this machine, or None."""
    options = find_options(name, **kwargs)
    return options[0] if options else None
