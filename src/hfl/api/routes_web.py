# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Ollama-compatible web_search + web_fetch endpoints (Phase 9 P0).

Contract (per https://docs.ollama.com/capabilities/web-search):

    POST /api/web_search
        body: {"query": str, "max_results": 1..10}
        → 200 {"results": [{"title","url","content"}, ...]}

    POST /api/web_fetch
        body: {"url": str}
        → 200 {"title": str, "content": str, "links": [str], "url": str}

Failures are reported as 400 with a curated generic message; the
full traceback stays in the server log. SSRF guards in
``web_fetch`` cover private IPs and the cloud-metadata endpoint.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from hfl.tools.web_fetch import WebFetchError, fetch
from hfl.tools.web_search import WebSearchError, search

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ollama"])


class WebSearchRequest(BaseModel):
    """Body for ``POST /api/web_search``."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Search query text.",
    )
    max_results: int = Field(
        5,
        ge=1,
        le=10,
        description="Cap on results returned (1-10, default 5).",
    )


class WebFetchRequest(BaseModel):
    """Body for ``POST /api/web_fetch``."""

    url: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Absolute URL to fetch (http/https only).",
    )


@router.post("/api/web_search")
async def api_web_search(req: WebSearchRequest) -> dict:
    """Search the web and return a list of ranked results.

    Backend is chosen by the ``HFL_WEB_SEARCH_BACKEND`` env var at
    server start: ``duckduckgo`` (default, free), ``tavily``,
    ``brave``, or ``serpapi``. API-keyed backends fail with 400 if
    their env var is missing.
    """
    try:
        return await search(req.query, req.max_results)
    except WebSearchError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("web_search crashed")
        raise HTTPException(status_code=500, detail="internal web_search error")


@router.post("/api/web_fetch")
async def api_web_fetch(req: WebFetchRequest) -> dict:
    """Fetch a URL and return extracted title / content / links.

    SSRF guard rejects private / loopback / link-local hosts before
    touching the network, matching Ollama's own posture.
    """
    try:
        return await fetch(req.url)
    except WebFetchError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("web_fetch crashed")
        raise HTTPException(status_code=500, detail="internal web_fetch error")


__all__ = ["router", "WebSearchRequest", "WebFetchRequest"]
