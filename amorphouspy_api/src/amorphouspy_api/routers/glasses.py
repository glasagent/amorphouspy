"""Glasses router — ``/glasses`` endpoints.

Read-only materials layer: a view over completed simulation jobs.
Use these endpoints to browse and search computed glass properties.

For job management (submitting, monitoring, cancelling jobs in any
stage), use the ``/jobs`` endpoints instead.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from amorphouspy_api.auth import verify_token
from amorphouspy_api.database import get_job_store
from amorphouspy_api.models import (
    AvailableStructure,
    Composition,
    GlassJobSummary,
    GlassListResponse,
    GlassLookupRequest,
    GlassPropertiesResponse,
    GlassSearchMatch,
    GlassSearchRequest,
    GlassSearchResponse,
    GlassSummary,
    _job_urls,
)
from amorphouspy_api.routers.jobs_helpers import (
    _analyses_list,
    find_close_matches,
    oxide_to_elemental_vector,
)
from amorphouspy_api.workflows import ANALYSES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/glasses", tags=["tool"], dependencies=[Depends(verify_token)])


@router.get("", response_model=GlassListResponse)
def list_glasses(
    tag: Annotated[str | None, Query(description="Filter to jobs with this tag")] = None,
) -> GlassListResponse:
    """List all compositions with completed simulation data.

    Returns per-composition summaries including individual job details
    (potential, tags, completed analyses).  Use the ``tag`` query parameter
    to restrict results to jobs carrying a specific tag.
    """
    store = get_job_store()
    rows = store.list_glasses(tag=tag)

    grouped: dict[str, list] = defaultdict(list)
    for r in rows:
        grouped[r["composition"]].append(r)

    glasses: list[GlassSummary] = []
    for comp_str, comp_jobs in grouped.items():
        job_summaries = [
            GlassJobSummary(
                job_id=r["job_id"],
                potential=r["potential"],
                tags=r["tags"],
                analyses=r["analyses"],
                completed_at=r["completed_at"].isoformat() if r["completed_at"] else None,
                urls=_job_urls(r["job_id"]),
            )
            for r in comp_jobs
        ]
        glasses.append(
            GlassSummary(
                composition=Composition.from_canonical(comp_str),
                n_jobs=len(comp_jobs),
                jobs=job_summaries,
            )
        )

    return GlassListResponse(glasses=glasses)


@router.post(":lookup", response_model=GlassPropertiesResponse)
def lookup_glass(body: GlassLookupRequest) -> GlassPropertiesResponse:
    """Get aggregated properties for a specific composition.

    Accepts a JSON body with a composition dict, e.g.::

        {"composition": {"SiO2": 70, "Na2O": 15, "CaO": 15}}
    """
    store = get_job_store()
    norm = body.composition.canonical
    jobs = store.search_by_composition(norm, statuses=["completed"])
    if not jobs:
        raise HTTPException(status_code=404, detail=f"No completed jobs for composition: {norm}")

    # Aggregate properties from the most recent completed job
    latest = jobs[0]
    result = latest.result_data or {}

    properties: dict[str, dict] = {}
    if result.get("structure_characterization"):
        properties["structure_characterization"] = {
            "data": result["structure_characterization"],
            "source_job": latest.job_id,
            "potential": latest.potential,
            "computed_at": (latest.completed_at.isoformat() if latest.completed_at else None),
        }

    # Collect available structures across all completed jobs
    available_structures = []
    for j in jobs:
        r = j.result_data or {}
        n_atoms = 0
        mq = r.get("melt_quench", {})
        if mq.get("final_structure"):
            from amorphouspy_api.models import validate_atoms

            atoms = validate_atoms(mq["final_structure"])
            n_atoms = len(atoms) if atoms else 0
        available_structures.append(
            AvailableStructure(
                job_id=j.job_id,
                potential=j.potential,
                n_atoms=n_atoms,
                visualization_url=_job_urls(j.job_id)["visualization"],
            )
        )

    # List analysis types not yet computed
    computed = set(properties.keys())
    missing = [t for t in ANALYSES if t not in computed]

    return GlassPropertiesResponse(
        composition=body.composition,
        properties=properties,
        available_structures=available_structures,
        missing=missing,
    )


@router.post(":search", response_model=GlassSearchResponse)
def search_glasses(body: GlassSearchRequest) -> GlassSearchResponse:
    """Search for completed glasses by composition similarity.

    Only includes completed jobs.  Returns exact composition matches
    first, then close matches within *threshold* Euclidean distance
    in elemental atom-fraction space, sorted by ascending distance.

    To search jobs in any lifecycle stage, use
    ``POST /jobs:search`` instead.
    """
    store = get_job_store()
    norm_comp = body.composition.canonical
    query_vec = oxide_to_elemental_vector(body.composition.root)

    # --- exact matches ---
    exact_jobs = store.search_by_composition(norm_comp, body.potential, statuses=["completed"])
    exact_ids = {j.job_id for j in exact_jobs}
    matches: list[GlassSearchMatch] = [
        GlassSearchMatch(
            job_id=j.job_id,
            composition=Composition.from_canonical(j.composition),
            potential=j.potential,
            tags=j.tags or [],
            analyses=_analyses_list(j),
            similarity=1.0,
            match_type="exact",
            distance=0.0,
            completed_at=j.completed_at.isoformat() if j.completed_at else None,
            visualization_url=_job_urls(j.job_id)["visualization"],
        )
        for j in exact_jobs
    ]

    # --- close matches (if threshold > 0) ---
    if body.threshold > 0:
        rows = store.list_completed_vectors(body.potential)
        scored = find_close_matches(
            query_vec,
            rows,
            exclude_ids=exact_ids,
            threshold=body.threshold,
            max_results=body.max_results,
        )
        # Batch-fetch tags for close matches
        close_ids = [job_id for _, job_id, *_ in scored]
        tags_map = store.get_tags_for_jobs(close_ids) if close_ids else {}

        for dist, job_id, comp, potential, req_data, completed_at in scored:
            sim = 1.0 / (1.0 + dist)
            analyses = [a.get("type", "structure_characterization") for a in (req_data or {}).get("analyses", [])]
            matches.append(
                GlassSearchMatch(
                    job_id=job_id,
                    composition=Composition.from_canonical(comp),
                    potential=potential,
                    tags=tags_map.get(job_id, []),
                    analyses=analyses,
                    similarity=round(sim, 4),
                    match_type="close",
                    distance=round(dist, 4),
                    completed_at=completed_at.isoformat() if completed_at else None,
                    visualization_url=_job_urls(job_id)["visualization"],
                )
            )

    # --- filter by tags (if requested) ---
    if body.tags:
        required = set(body.tags)
        matches = [m for m in matches if required <= set(m.tags)]

    return GlassSearchResponse(matches=matches)
