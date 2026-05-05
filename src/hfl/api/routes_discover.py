# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""V4 ``GET /api/discover`` — Hub-backed model catalogue.

Wraps :mod:`hfl.hub.discovery` with HTTP plumbing: query-string
validation via FastAPI, on-disk cache lookup, registry join for
``locally_available``, and a stable JSON envelope shaped after
``/api/tags``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from hfl.hub.discovery import (
    DiscoveryCache,
    DiscoveryEntry,
    DiscoveryQuery,
    DiscoveryResult,
    search_hub,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["HFL Beyond"])


def _build_cache() -> DiscoveryCache:
    """Locate the discovery cache under ``HFL_HOME/cache/``."""
    from hfl.config import config as hfl_config

    return DiscoveryCache(hfl_config.cache_dir / "discovery.json")


def _annotate_local_availability(entries: list[DiscoveryEntry]) -> None:
    """Join Hub results against the local registry by ``repo_id``.

    Mutates entries in place. Failure to consult the registry is
    non-fatal — the field stays ``False`` and the client sees an
    accurate "not local" indicator.
    """
    try:
        from hfl.core.container import get_registry

        registry = get_registry()
        local_repo_ids = {
            (m.repo_id or "").strip() for m in registry.list_all() if (m.repo_id or "").strip()
        }
    except Exception:
        local_repo_ids = set()

    for entry in entries:
        entry.locally_available = entry.repo_id in local_repo_ids


@router.get(
    "/api/discover",
    response_model=None,
    summary="Discover models on the HuggingFace Hub",
    responses={
        200: {"description": "Discovery results"},
        400: {"description": "Invalid filter combination"},
        503: {"description": "Hub unavailable"},
    },
)
async def api_discover(
    q: str | None = Query(default=None, max_length=256),
    family: str | None = Query(default=None, max_length=64),
    task: str | None = Query(default=None, max_length=64),
    quantization: str | None = Query(default=None, max_length=32),
    multimodal: bool = Query(default=False),
    min_likes: int = Query(default=0, ge=0, le=1_000_000),
    min_downloads: int = Query(default=0, ge=0, le=1_000_000_000),
    license: str | None = Query(default=None, max_length=64),
    gated: bool | None = Query(default=None),
    page_size: int = Query(default=30, ge=1, le=100),
    refresh: bool = Query(default=False, description="Bypass the on-disk cache"),
) -> dict[str, Any]:
    """V4: filter the HF Hub catalogue by capability/popularity.

    The endpoint is read-only and does not require authentication
    (gated repos still need ``HF_TOKEN`` to actually load — but
    they're visible in discovery so users learn they exist).
    """
    query = DiscoveryQuery(
        q=q,
        family=family,
        task=task,
        quantization=quantization,
        multimodal=multimodal,
        min_likes=min_likes,
        min_downloads=min_downloads,
        license=license,
        gated=gated,
        page_size=page_size,
    )

    cache = _build_cache()
    cached_hit = None if refresh else cache.get(query)
    if cached_hit is not None:
        _annotate_local_availability(cached_hit)
        result = DiscoveryResult(
            query=query.__dict__.copy(),
            total=len(cached_hit),
            entries=cached_hit,
            cached=True,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )
        return _serialise(result)

    try:
        entries = search_hub(query)
    except Exception as exc:
        logger.exception("Hub discovery failed")
        raise HTTPException(status_code=503, detail=f"Hub unavailable: {exc}")

    cache.put(query, entries)
    _annotate_local_availability(entries)

    result = DiscoveryResult(
        query=query.__dict__.copy(),
        total=len(entries),
        entries=entries,
        cached=False,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )
    return _serialise(result)


def _serialise(result: DiscoveryResult) -> dict[str, Any]:
    """Render a ``DiscoveryResult`` as a JSON-friendly dict.

    FastAPI would auto-serialise the dataclass, but the on-disk
    cache stores a slightly different shape; centralising the
    rendering keeps both paths consistent.
    """
    from dataclasses import asdict

    return {
        "query": result.query,
        "total": result.total,
        "cached": result.cached,
        "fetched_at": result.fetched_at,
        "entries": [asdict(e) for e in result.entries],
    }
