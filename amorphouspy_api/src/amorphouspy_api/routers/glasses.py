"""Glasses router — ``/glasses`` endpoints.

Read-only materials layer: a view over completed jobs.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from amorphouspy_api.database import get_job_store
from amorphouspy_api.models import (
    AvailableStructure,
    Composition,
    GlassListResponse,
    GlassLookupRequest,
    GlassPropertiesResponse,
    GlassSummary,
)
from amorphouspy_api.workflows import ANALYSES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/glasses", tags=["tool"])


@router.get("", response_model=GlassListResponse)
def list_glasses() -> GlassListResponse:
    """List all compositions with completed simulation data."""
    store = get_job_store()
    rows = store.list_compositions()
    return GlassListResponse(
        glasses=[
            GlassSummary(
                composition=Composition.from_canonical(r["composition"]),
                n_jobs=r["n_jobs"],
            )
            for r in rows
        ],
    )


@router.post(":lookup", response_model=GlassPropertiesResponse)
def lookup_glass(request: GlassLookupRequest) -> GlassPropertiesResponse:
    """Get aggregated properties for a specific composition.

    Accepts a JSON body with a composition dict, e.g.::

        {"composition": {"SiO2": 70, "Na2O": 15, "CaO": 15}}
    """
    store = get_job_store()
    norm = request.composition.canonical
    jobs = store.search_by_composition(norm)
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
            )
        )

    # List analysis types not yet computed
    computed = set(properties.keys())
    missing = [t for t in ANALYSES if t not in computed]

    return GlassPropertiesResponse(
        composition=request.composition,
        properties=properties,
        available_structures=available_structures,
        missing=missing,
    )
