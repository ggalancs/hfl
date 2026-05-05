# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""HuggingFace Hub discovery — V4 differentiator.

Ollama's local catalogue is ~3K curated entries with no metadata.
The HF Hub is 1.5M+ models with structured metadata (likes,
downloads, family, license, gated status, modalities). HFL exposes
that catalogue through ``GET /api/discover`` and ``GET /api/recommend``
so a client can ask "what's relevant for me right now?" without
leaving the server.

This module is the lower layer: pure Hub access + result shaping.
The HTTP route lives in ``hfl/api/routes_discover.py``.

Design constraints:

- The Hub rate-limits unauthenticated requests; calls are cached
  on-disk for 5 minutes (``DiscoveryCache``) so chained CLI calls
  don't get 429-ed.
- ``HfApi.list_models`` returns paginated lazy iterators; we cap at
  ``page_size`` to avoid hammering the network.
- "Already local" annotation joins HF results against the local
  registry by ``repo_id`` so the client renders the right action
  ("download" vs "load").
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from huggingface_hub import HfApi, ModelInfo

logger = logging.getLogger(__name__)

__all__ = [
    "DiscoveryQuery",
    "DiscoveryResult",
    "DiscoveryCache",
    "search_hub",
    "format_size_human",
]


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveryQuery:
    """Filters accepted by ``GET /api/discover``.

    Names match the Hub's own filter vocabulary where possible
    (``library``, ``task``) so a client that already speaks HF can
    map its UI 1:1 to ours. ``family`` is HFL-specific (mapped to
    Hub tags via :func:`_family_to_tag`).
    """

    q: str | None = None
    family: str | None = None  # llama, qwen, gemma, mistral, phi
    task: str | None = None  # text-generation, embeddings, etc.
    quantization: str | None = None  # gguf, awq, gptq, mlx
    multimodal: bool = False
    min_likes: int = 0
    min_downloads: int = 0
    license: str | None = None
    gated: bool | None = None  # None = include both
    page_size: int = 30


@dataclass
class DiscoveryEntry:
    """One model row in a discovery response.

    Designed to be JSON-serialisable directly (no special types).
    The ``locally_available`` flag is filled in by the route layer
    after the join against the local registry.
    """

    repo_id: str
    likes: int
    downloads: int
    last_modified: str | None
    pipeline_tag: str | None
    library: str | None
    license: str | None
    gated: bool
    tags: list[str] = field(default_factory=list)
    family: str | None = None
    quantization: str | None = None
    parameter_estimate_b: float | None = None
    locally_available: bool = False


@dataclass
class DiscoveryResult:
    """Top-level envelope for ``GET /api/discover``."""

    query: dict[str, Any]
    total: int
    entries: list[DiscoveryEntry]
    cached: bool = False
    fetched_at: str | None = None


# ---------------------------------------------------------------------------
# Family / quantisation classification
# ---------------------------------------------------------------------------


_FAMILY_TAGS: dict[str, list[str]] = {
    # Map HFL-friendly family names to Hub tag substrings. The Hub
    # uses inconsistent casing/punctuation across orgs; these are
    # checked case-insensitively against the tag list.
    #
    # Order matters: more specific names FIRST so a repo path like
    # ``mistralai/Mixtral-8x7B`` resolves to ``mixtral`` (not
    # ``mistral`` — the namespace would match either, but mixtral is
    # the actual family).
    "mixtral": ["mixtral"],
    "command-r": ["command-r"],
    "deepseek": ["deepseek"],
    "llama": ["llama"],
    "qwen": ["qwen"],
    "gemma": ["gemma"],
    "mistral": ["mistral"],
    "phi": ["phi"],
    "yi": ["yi"],
    "falcon": ["falcon"],
}


_QUANT_KEYWORDS: dict[str, list[str]] = {
    "gguf": ["gguf", "ggml"],
    "awq": ["awq"],
    "gptq": ["gptq"],
    "mlx": ["mlx", "-mlx-", "mlx-community"],
    "exl2": ["exl2"],
    "fp8": ["fp8"],
    "int4": ["int4", "4bit"],
    "int8": ["int8", "8bit"],
}


_MULTIMODAL_KEYWORDS: tuple[str, ...] = (
    "vision",
    "vl",
    "multimodal",
    "image-text",
    "image-to-text",
)


def _family_for(repo_id: str, tags: Iterable[str]) -> str | None:
    """Identify the model family from repo id + tag list.

    Tags are stronger evidence than repo id (the Hub adds them when
    the model card declares ``base_model``); fall back to the id
    substring for unscoped repos.
    """
    tag_pool = " ".join(tags).lower()
    full = (repo_id + " " + tag_pool).lower()
    for family, needles in _FAMILY_TAGS.items():
        if any(n in full for n in needles):
            return family
    return None


def _quantization_for(repo_id: str, tags: Iterable[str]) -> str | None:
    full = (repo_id + " " + " ".join(tags)).lower()
    for quant, needles in _QUANT_KEYWORDS.items():
        if any(n in full for n in needles):
            return quant
    return None


def _is_multimodal(tags: Iterable[str], pipeline_tag: str | None) -> bool:
    if pipeline_tag and any(kw in pipeline_tag for kw in _MULTIMODAL_KEYWORDS):
        return True
    pool = " ".join(tags).lower()
    return any(kw in pool for kw in _MULTIMODAL_KEYWORDS)


def _parameter_estimate_b(repo_id: str, tags: Iterable[str]) -> float | None:
    """Best-effort parameter count from id / tags.

    Hub model names usually advertise size: ``Llama-3.2-1B``,
    ``Qwen2.5-7B``, ``Mixtral-8x7B``. Returns billions as float, or
    ``None`` when no signal. Tolerant to variants like ``8x7B``
    (Mixtral) — falls back to the second number.
    """
    import re

    full = repo_id + " " + " ".join(tags)
    # Match ``<N>B`` where N may carry a decimal.
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])", full)
    if not matches:
        return None
    try:
        # Prefer the LAST match — repo names like ``Llama-3.1-8B-Instruct``
        # have the size right before the variant tag.
        return float(matches[-1])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Hub query
# ---------------------------------------------------------------------------


def _matches_filters(entry: DiscoveryEntry, query: DiscoveryQuery) -> bool:
    """Apply HFL-specific post-filters that Hub doesn't natively express.

    The Hub's ``list_models`` accepts ``filter=`` but its grammar is
    limited; we filter by likes/downloads/family/quantization in
    Python after fetching the page.
    """
    if entry.likes < query.min_likes:
        return False
    if entry.downloads < query.min_downloads:
        return False
    if query.family and entry.family != query.family:
        return False
    if query.quantization and entry.quantization != query.quantization:
        return False
    if query.gated is not None and entry.gated != query.gated:
        return False
    if query.license and (entry.license or "").lower() != query.license.lower():
        return False
    return True


def _entry_from_model_info(info: "ModelInfo") -> DiscoveryEntry:
    """Convert a ``huggingface_hub.ModelInfo`` to our typed shape."""
    tags = list(getattr(info, "tags", None) or [])
    repo_id = info.id  # type: ignore[attr-defined]

    pipeline_tag = getattr(info, "pipeline_tag", None)
    family = _family_for(repo_id, tags)
    quant = _quantization_for(repo_id, tags)
    is_mm = _is_multimodal(tags, pipeline_tag)
    if is_mm and "multimodal" not in tags:
        tags.append("multimodal")

    last_modified = getattr(info, "last_modified", None)
    last_modified_str = last_modified.isoformat() if last_modified is not None else None

    card = getattr(info, "card_data", None)
    license_value = None
    if card is not None:
        license_value = getattr(card, "license", None)

    return DiscoveryEntry(
        repo_id=repo_id,
        likes=int(getattr(info, "likes", 0) or 0),
        downloads=int(getattr(info, "downloads", 0) or 0),
        last_modified=last_modified_str,
        pipeline_tag=pipeline_tag,
        library=getattr(info, "library_name", None),
        license=license_value,
        gated=bool(getattr(info, "gated", False)),
        tags=tags,
        family=family,
        quantization=quant,
        parameter_estimate_b=_parameter_estimate_b(repo_id, tags),
    )


def search_hub(
    query: DiscoveryQuery,
    *,
    api: "HfApi | None" = None,
) -> list[DiscoveryEntry]:
    """Run a Hub discovery query.

    ``api`` injection lets tests pass a fake without monkeypatching
    the import. Production callers leave it ``None`` and we
    construct a fresh ``HfApi`` per call.
    """
    if api is None:
        from huggingface_hub import HfApi as _HfApi

        api = _HfApi()

    list_kwargs: dict[str, Any] = {
        "limit": max(query.page_size * 3, 30),  # over-fetch then filter
        "sort": "downloads",
        "fetch_config": False,
    }
    if query.q:
        list_kwargs["search"] = query.q
    if query.task:
        list_kwargs["pipeline_tag"] = query.task
    if query.family:
        list_kwargs["filter"] = _FAMILY_TAGS.get(query.family, [query.family])[0]

    raw = api.list_models(**list_kwargs)

    matched: list[DiscoveryEntry] = []
    for info in raw:
        try:
            entry = _entry_from_model_info(info)
        except Exception:  # pragma: no cover — defensive
            continue
        if query.multimodal and not _is_multimodal(entry.tags, entry.pipeline_tag):
            continue
        if not _matches_filters(entry, query):
            continue
        matched.append(entry)
        if len(matched) >= query.page_size:
            break

    return matched


def format_size_human(b: float | None) -> str:
    """Render a parameter count as ``1.5B`` / ``70B`` / ``-``."""
    if b is None:
        return "-"
    if b >= 100:
        return f"{int(b)}B"
    if b == int(b):
        return f"{int(b)}B"
    return f"{b:.1f}B"


# ---------------------------------------------------------------------------
# On-disk LRU cache
# ---------------------------------------------------------------------------


class DiscoveryCache:
    """Filesystem cache for Hub discovery results.

    The Hub rate-limits unauthenticated traffic; a CLI user typing
    ``hfl discover ...`` repeatedly hits the same query. Cache keyed
    by the JSON-serialised query dict, TTL 5 minutes, single file
    in ``HFL_HOME/cache/discovery.json``.
    """

    def __init__(self, path: Path, ttl_seconds: int = 300) -> None:
        self._path = path
        self._ttl = ttl_seconds

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            with self._path.open() as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            # Corrupt cache shouldn't break the request — start fresh.
            logger.warning("Discovery cache at %s is corrupt; resetting", self._path)
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(data, f)
        tmp.replace(self._path)

    @staticmethod
    def _key(query: DiscoveryQuery) -> str:
        return json.dumps(asdict(query), sort_keys=True)

    def get(self, query: DiscoveryQuery) -> list[DiscoveryEntry] | None:
        data = self._load()
        cell = data.get(self._key(query))
        if cell is None:
            return None
        if time.time() - cell["t"] > self._ttl:
            return None
        return [DiscoveryEntry(**row) for row in cell["entries"]]

    def put(self, query: DiscoveryQuery, entries: list[DiscoveryEntry]) -> None:
        data = self._load()
        # Bound the cache: keep only the 32 most recent keys.
        if len(data) > 32:
            stale = sorted(data.items(), key=lambda kv: kv[1]["t"])[: len(data) - 32]
            for stale_key, _ in stale:
                data.pop(stale_key, None)

        data[self._key(query)] = {
            "t": time.time(),
            "entries": [asdict(e) for e in entries],
        }
        self._save(data)
