"""Glasses router — ``/glasses`` endpoints.

Read-only materials layer: a view over completed jobs.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from amorphouspy_api.composition import normalize_composition
from amorphouspy_api.database import get_job_store
from amorphouspy_api.models import (
    AvailableStructure,
    GlassListResponse,
    GlassPropertiesResponse,
    GlassSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/glasses", tags=["tool"])

ALL_ANALYSIS_TYPES = ["structure", "elastic", "viscosity", "cte"]


@router.get("", response_model=GlassListResponse | GlassPropertiesResponse)
def get_glasses(
    composition: Annotated[str | None, Query(description="Filter by composition")] = None,
) -> GlassListResponse | GlassPropertiesResponse:
    """List glasses or get properties for a specific composition.

    * ``GET /glasses`` → list all compositions with completed data.
    * ``GET /glasses?composition=SiO2 70 - Na2O 30`` → aggregated properties.
    """
    store = get_job_store()

    if composition is None:
        rows = store.list_compositions()
        return GlassListResponse(
            glasses=[GlassSummary(**r) for r in rows],
        )

    # Specific composition lookup
    norm = normalize_composition(composition)
    jobs = store.search_by_composition(norm)
    if not jobs:
        raise HTTPException(status_code=404, detail=f"No completed jobs for composition: {norm}")

    # Aggregate properties from the most recent completed job
    latest = jobs[0]
    result = latest.result_data or {}

    properties: dict[str, dict] = {}
    if result.get("structural_analysis"):
        properties["structure"] = {
            "data": result["structural_analysis"],
            "source_job": latest.job_id,
            "potential": latest.potential,
            "computed_at": latest.completed_at.isoformat() if latest.completed_at else None,
        }

    # Collect available structures across all completed jobs
    available_structures = []
    for j in jobs:
        r = j.result_data or {}
        n_atoms = 0
        if r.get("final_structure"):
            # Try to get atom count from the structure
            from amorphouspy_api.models import validate_atoms

            atoms = validate_atoms(r["final_structure"])
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
    missing = [t for t in ALL_ANALYSIS_TYPES if t not in computed]

    return GlassPropertiesResponse(
        composition=norm,
        properties=properties,
        available_structures=available_structures,
        missing=missing,
    )
