"""Jobs router — ``/jobs`` endpoints.

Endpoints:
  POST /jobs          - submit a new simulation
  POST /jobs:search   - find cached / similar jobs
  GET  /jobs/{id}     - poll job status
  POST /jobs/{id}:cancel - cancel a running job
  GET  /jobs/{id}/results            - all analysis results
  GET  /jobs/{id}/results/{analysis} - single analysis result
  GET  /jobs/{id}/structure          - export quenched structure
  GET  /jobs/{id}/visualize          - interactive HTML visualization
"""

from __future__ import annotations

import logging
from io import StringIO
from typing import Annotated
from uuid import uuid4

from ase.io import write as ase_write
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

from amorphouspy_api.database import Job, get_job_store
from amorphouspy_api.models import (
    Composition,
    JobCreatedResponse,
    JobResultsResponse,
    JobSearchMatch,
    JobSearchRequest,
    JobSearchResponse,
    JobStatus,
    JobStatusResponse,
    JobSubmission,
    validate_atoms,
)
from amorphouspy_api.routers.jobs_helpers import (
    _analyses_list,
    _initial_progress,
    _iso_now,
    _job_hash,
    _progress_from_dict,
    _submit_to_executor,
    _update_from_resolved,
    build_visualization_context,
    find_close_matches,
    oxide_to_elemental_vector,
    refresh_job_from_cache,
)
from amorphouspy_api.workflows import ANALYSES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["tool"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=JobCreatedResponse)
def submit_job(submission: JobSubmission) -> JobCreatedResponse:
    """Submit a new simulation job.

    The server resolves the dependency DAG internally.
    If an identical job already completed, the cached result is returned.
    """
    store = get_job_store()
    norm_comp = submission.composition.canonical
    req_hash = _job_hash(submission, norm_comp)

    # Check for cached result
    cached = store.find_completed_by_hash(req_hash)
    if cached:
        logger.info("Returning cached job %s", cached.job_id)
        return JobCreatedResponse(
            id=cached.job_id,
            status=JobStatus(cached.status),
            composition=Composition.from_canonical(cached.composition),
            potential=cached.potential,
            created_at=(cached.created_at.isoformat() if cached.created_at else _iso_now()),
        )

    # Create new job record
    job_id = str(uuid4())

    job = Job(
        job_id=job_id,
        request_hash=req_hash,
        composition=norm_comp,
        potential=submission.potential.value,
        status="pending",
        request_data=submission.model_dump(),
        progress=_initial_progress(submission),
    )
    store.create_job(job)
    logger.info("Created job %s (hash=%s)", job_id, req_hash)

    # Submit to executor
    try:
        resolved = _submit_to_executor(submission, job_id, req_hash)
        _update_from_resolved(job_id, resolved, submission)
    except Exception:
        logger.exception("Failed to submit job %s", job_id)
        store.update_job(
            job_id,
            status="failed",
            errors={"submission": "Failed to start executor"},
        )

    # Re-read to get final state
    final = store.get_job(job_id)
    return JobCreatedResponse(
        id=job_id,
        status=JobStatus(final.status) if final else JobStatus.PENDING,
        composition=Composition.from_canonical(norm_comp),
        potential=submission.potential,
        created_at=(final.created_at.isoformat() if final and final.created_at else _iso_now()),
    )


@router.post(":search", response_model=JobSearchResponse)
def search_jobs(request: JobSearchRequest) -> JobSearchResponse:
    """Search for existing completed / running jobs matching a spec.

    Returns exact composition matches first (similarity=1.0), then
    close matches within *threshold* Euclidean distance in elemental
    atom-fraction space, sorted by ascending distance.
    """
    store = get_job_store()
    norm_comp = request.composition.canonical
    query_vec = oxide_to_elemental_vector(request.composition.root)

    # --- exact matches ---
    exact_jobs = store.search_by_composition(norm_comp, request.potential)
    exact_ids = {j.job_id for j in exact_jobs}
    matches = [
        JobSearchMatch(
            job_id=j.job_id,
            composition=Composition.from_canonical(j.composition),
            potential=j.potential,
            analyses=_analyses_list(j),
            similarity=1.0,
            match_type="exact",
            distance=0.0,
            completed_at=j.completed_at.isoformat() if j.completed_at else None,
        )
        for j in exact_jobs
    ]

    # --- close matches (if threshold > 0) ---
    if request.threshold > 0:
        rows = store.list_completed_vectors(request.potential)
        scored = find_close_matches(
            query_vec,
            rows,
            exclude_ids=exact_ids,
            threshold=request.threshold,
            max_results=request.max_results,
        )
        for dist, job_id, comp, potential, req_data, completed_at in scored:
            sim = 1.0 / (1.0 + dist)
            analyses = [a.get("type", "structure_characterization") for a in (req_data or {}).get("analyses", [])]
            matches.append(
                JobSearchMatch(
                    job_id=job_id,
                    composition=Composition.from_canonical(comp),
                    potential=potential,
                    analyses=analyses,
                    similarity=round(sim, 4),
                    match_type="close",
                    distance=round(dist, 4),
                    completed_at=completed_at.isoformat() if completed_at else None,
                )
            )

    return JobSearchResponse(matches=matches)


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Poll job status with per-step progress."""
    store = get_job_store()
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # If still running, try to refresh from executor cache
    if job.status == "running":
        refresh_job_from_cache(job)
        job = store.get_job(job_id)

    return JobStatusResponse(
        id=job.job_id,
        status=JobStatus(job.status),
        composition=Composition.from_canonical(job.composition),
        potential=job.potential,
        progress=_progress_from_dict(job.progress),
        errors=job.errors or {},
        created_at=job.created_at.isoformat() if job.created_at else _iso_now(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


@router.post("/{job_id}:cancel")
def cancel_job(job_id: str) -> JobStatusResponse:
    """Cancel a running job."""
    store = get_job_store()
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in ("pending", "running"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel job in status '{job.status}'")

    # Mark pending/running steps as cancelled
    progress = dict(job.progress or {})
    for step, step_status in progress.items():
        if step_status in ("pending", "running"):
            progress[step] = "cancelled"

    store.update_job(job_id, status="cancelled", progress=progress)
    job = store.get_job(job_id)

    return JobStatusResponse(
        id=job.job_id,
        status=JobStatus(job.status),
        composition=Composition.from_canonical(job.composition),
        potential=job.potential,
        progress=_progress_from_dict(job.progress),
        errors=job.errors or {},
        created_at=job.created_at.isoformat() if job.created_at else _iso_now(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


@router.get("/{job_id}/results", response_model=JobResultsResponse)
def get_job_results(job_id: str) -> JobResultsResponse:
    """All completed analysis results for a job."""
    store = get_job_store()
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = job.result_data
    if not result:
        raise HTTPException(status_code=404, detail="No results available yet")

    # Collect results for all analysis types
    analyses = {key: result[key] for key in ANALYSES if key in result}

    return JobResultsResponse(
        job_id=job.job_id,
        composition=Composition.from_canonical(job.composition),
        analyses=analyses,
    )


@router.get("/{job_id}/results/{analysis}")
def get_single_result(job_id: str, analysis: str) -> dict:
    """Results for a single analysis type."""
    store = get_job_store()
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = job.result_data or {}
    key = analysis
    if key not in result:
        raise HTTPException(status_code=404, detail=f"No results for analysis '{analysis}'")

    return {"job_id": job.job_id, analysis: result[key]}


@router.get("/{job_id}/structure")
def get_structure(
    job_id: str,
    fmt: Annotated[
        str,
        Query(alias="format", description="Export format: xyz, cif, poscar, extxyz"),
    ] = "xyz",
) -> Response:
    """Export the final quenched structure."""
    store = get_job_store()
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = job.result_data or {}
    mq = result.get("melt_quench", {})
    raw_structure = mq.get("final_structure")
    if not raw_structure:
        raise HTTPException(status_code=404, detail="No structure available")

    atoms = validate_atoms(raw_structure)
    if atoms is None:
        raise HTTPException(status_code=404, detail="Could not parse structure")

    fmt_map = {
        "xyz": ("chemical/x-xyz", "xyz"),
        "extxyz": ("chemical/x-xyz", "extxyz"),
        "cif": ("chemical/x-cif", "cif"),
        "poscar": ("text/plain", "vasp"),
    }
    if fmt not in fmt_map:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")

    content_type, ase_fmt = fmt_map[fmt]
    buf = StringIO()
    ase_write(buf, atoms, format=ase_fmt)

    return Response(content=buf.getvalue(), media_type=content_type)


@router.get("/{job_id}/visualize", response_class=HTMLResponse)
def visualize_job(job_id: str) -> HTMLResponse:
    """Interactive HTML visualization of completed results."""
    from pathlib import Path

    from fastapi.templating import Jinja2Templates

    store = get_job_store()
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Refresh running jobs so we pick up newly-completed steps
    if job.status == "running":
        refresh_job_from_cache(job)
        job = store.get_job(job_id)

    if job.status == "pending":
        raise HTTPException(
            status_code=400,
            detail="Job has not started yet.",
        )

    result_data = job.result_data or {}

    if not result_data:
        raise HTTPException(status_code=404, detail="No results available yet")

    try:
        context = build_visualization_context(
            job_id,
            result_data,
            request_hash=job.request_hash,
            request_data=job.request_data,
        )
        context["job_status"] = job.status
        context["progress"] = job.progress or {}

        template_dir = Path(__file__).parent.parent / "templates"
        templates = Jinja2Templates(directory=str(template_dir))
        html_content = templates.get_template("results.html").render(context)

        return HTMLResponse(content=html_content)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating visualisation for job %s", job_id)
        raise HTTPException(status_code=500, detail=f"Error generating visualisation: {e!s}") from e
