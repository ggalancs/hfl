# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""V4 F5 — ``GET /api/draft/recommend?model=<repo>``.

Returns a draft-model suggestion for speculative decoding against
the given target. Read-only — no mutation of the registry.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from hfl.hub.draft_picker import pick_draft_for

logger = logging.getLogger(__name__)

router = APIRouter(tags=["HFL Beyond"])


@router.get(
    "/api/draft/recommend",
    response_model=None,
    summary="Recommend a draft model for speculative decoding",
    responses={
        200: {"description": "Draft pick or null"},
        400: {"description": "Missing target model"},
    },
)
async def api_draft_recommend(
    model: str = Query(..., min_length=1, max_length=256),
    max_ratio: float = Query(default=0.25, gt=0.0, le=1.0),
) -> dict[str, Any]:
    """Pick a draft for ``model``.

    Output::

        {
            "target": "meta-llama/Llama-3.1-70B-Instruct",
            "pick": {
                "repo_id": "meta-llama/Llama-3.2-1B-Instruct",
                "family": "llama",
                "parameter_estimate_b": 1.0,
                "quantization": null,
                "rationale": "smaller sibling (1B vs 70B target)"
            }
        }

    ``pick`` is ``null`` when no candidate was found (rare — the
    canonical fallbacks cover the major families).
    """
    try:
        pick = pick_draft_for(model, max_ratio=max_ratio)
    except Exception as exc:  # pragma: no cover — Hub failure is upstream
        logger.exception("draft recommender failed")
        raise HTTPException(status_code=503, detail=f"Hub unavailable: {exc}")

    return {
        "target": model,
        "pick": asdict(pick) if pick is not None else None,
    }
