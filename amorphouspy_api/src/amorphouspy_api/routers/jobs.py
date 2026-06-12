"""Jobs router — ``/jobs`` endpoints.

Job management endpoints for submitting, monitoring, and retrieving
simulation jobs in any stage of their lifecycle (pending, running,
completed, failed, cancelled).

For querying completed simulation results by composition similarity,
use the ``/glasses`` endpoints instead.

Endpoints:
  POST /jobs          - submit a new simulation
  POST /jobs:search   - search jobs across all statuses
  GET  /jobs/{id}     - poll job status
  POST /jobs/{id}:cancel - cancel a running job
  GET  /jobs/{id}/settings          - original submission settings
  GET  /jobs/{id}/results            - all analysis results
  GET  /jobs/{id}/results/{analysis} - single analysis result
  GET  /jobs/{id}/structure          - export quenched structure
  GET  /jobs/{id}/visualize          - interactive HTML visualization
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from amorphouspy.fabrication import extract_composition
from ase.io import write as ase_write
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

from amorphouspy_api.auth import verify_token
from amorphouspy_api.database import Job, get_job_store
from amorphouspy_api.models import (
    Composition,
    JobCreatedResponse,
    JobResultsResponse,
    JobSearchMatch,
    JobSearchRequest,
    JobSearchResponse,
    JobSettingsResponse,
    JobStatus,
    JobStatusResponse,
    JobSubmission,
    RerunMode,
    TagsResponse,
    TagsUpdate,
    _job_urls,
    validate_atoms,
)
from amorphouspy_api.pipeline import ANALYSES
from amorphouspy_api.routers.jobs_helpers import (
    _analyses_list,
    _initial_progress,
    _iso_now,
    _job_hash,
    _progress_from_dict,
    _submit_to_executor,
    _update_from_resolved,
    build_visualization_context,
    refresh_job_from_cache,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["tool"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_executor_cache(request_hash: str, *, failed_only: bool = False) -> None:
    """Delete executorlib HDF5 cache files for *request_hash*.

    Args:
        request_hash: The cache key prefix.
        failed_only: When ``True``, only delete ``_o.h5`` files whose stored
            result is an error (plus the corresponding ``_i.h5``).  Successful
            outputs are preserved so executorlib can reuse them.
    """
    from amorphouspy_api.config import MELTQUENCH_PROJECT_DIR

    cache_dir = Path(MELTQUENCH_PROJECT_DIR)
    if not cache_dir.is_dir():
        return

    if not failed_only:
        for f in cache_dir.glob(f"{request_hash}*.h5"):
            logger.info("Removing cached file %s (force re-run)", f.name)
            f.unlink(missing_ok=True)
        return

    # Selective cleanup: only remove output files that contain errors.
    import h5py

    for f in sorted(cache_dir.glob(f"{request_hash}*_o.h5")):
        try:
            with h5py.File(f, "r") as hdf:
                if "output" in hdf:
                    continue  # successful — keep it
        except Exception:
            logger.debug("Could not read %s, treating as failed", f.name)
        logger.info("Removing failed cache file %s (retry)", f.name)
        f.unlink(missing_ok=True)
        # Also remove the matching _i.h5 so executorlib re-submits this step
        i_file = f.with_name(f.name.replace("_o.h5", "_i.h5"))
        if i_file.exists():
            logger.info("Removing input file %s (retry)", i_file.name)
            i_file.unlink(missing_ok=True)


def _composition_elements(composition: dict[str, float]) -> set[str]:
    """Extract element symbols from a composition dict."""
    from amorphouspy.fabrication import parse_formula

    elements: set[str] = set()
    for oxide in composition:
        elements.update(parse_formula(oxide))
    return elements


def _validate_or_select_potential(
    submission: JobSubmission,
) -> None:
    """Validate or auto-select the potential for *submission*.

    If the user explicitly chose a potential that doesn't support the
    composition, raise 422 with suggestions.  If the default (pmmcs) was
    kept and it doesn't fit, silently pick the best compatible one.
    """
    from amorphouspy.lammps.potentials.potential import (
        compatible_potentials,
        get_supported_elements,
        select_potential,
    )

    elements = _composition_elements(submission.composition.root)
    potential = submission.potential

    supported = get_supported_elements(potential)
    unsupported = elements - supported
    if not unsupported:
        return

    # Default was kept → auto-select the best compatible potential
    if submission.potential == "pmmcs":
        best = select_potential(elements)
        if best is not None:
            logger.info(
                "Auto-selected '%s' potential (pmmcs missing %s)",
                best,
                sorted(unsupported),
            )
            submission.potential = best
            return

    # Explicit choice or nothing works at all → 422
    compat = compatible_potentials(elements)
    hint = f" Compatible potentials: {compat}." if compat else ""
    raise HTTPException(
        status_code=422,
        detail=(f"The '{potential}' potential does not support element(s): {sorted(unsupported)}.{hint}"),
    )


def _return_cached_job(cached: Job, submission: JobSubmission, store) -> JobCreatedResponse:
    """Return an already-completed job, merging any new tags."""
    logger.info("Returning cached job %s", cached.job_id)
    if submission.tags:
        existing = set(cached.tags or [])
        merged = sorted(existing | set(submission.tags))
        if merged != sorted(existing):
            store.update_job(cached.job_id, tags=merged)
    return JobCreatedResponse(
        id=cached.job_id,
        status=JobStatus(cached.status),
        composition=Composition.from_canonical(cached.composition),
        potential=cached.potential,
        tags=sorted(set(cached.tags or []) | set(submission.tags)),
        created_at=(cached.created_at.isoformat() if cached.created_at else _iso_now()),
        urls=_job_urls(cached.job_id),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=JobCreatedResponse, dependencies=[Depends(verify_token)])
def submit_job(
    submission: JobSubmission,
    rerun: Annotated[
        RerunMode | None,
        Query(
            description=("Skip and delete cache for this job ('all' - all steps, 'failed' - failed steps only). "),
        ),
    ] = None,
) -> JobCreatedResponse:
    """Submit a new simulation job.

    The server resolves the dependency DAG internally.
    If an identical job already completed, the cached result is returned
    by default.
    """
    # Validate and normalise composition so downstream code always sees
    # fractions that sum to exactly 1.0 (the pipeline uses the default
    # tight tolerance of 0.001).
    try:
        normalised = extract_composition(submission.composition.root, tolerance=0.03)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Replace with rescaled mol-% (fractions → percentages summing to 100)
    submission.composition = Composition({ox: frac * 100 for ox, frac in normalised.items()})

    # Validate potential / auto-select if default doesn't fit
    _validate_or_select_potential(submission)

    # Resolve target density: predict from model if not supplied, fail fast
    # if the model cannot handle the composition.
    if submission.simulation.target_density is None:
        from amorphouspy.fabrication.density import get_glass_density_from_model

        try:
            submission.simulation.target_density = get_glass_density_from_model(
                submission.composition.root,
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Cannot estimate density for the given composition: {exc}. "
                    "Please provide an explicit 'target_density' in the simulation parameters."
                ),
            ) from exc

    store = get_job_store()
    norm_comp = submission.composition.canonical
    req_hash = _job_hash(submission, norm_comp)

    # Check for cached result (skipped when rerun is set)
    if rerun is None:
        cached = store.find_completed_by_hash(req_hash)
        if cached:
            return _return_cached_job(cached, submission, store)

    # Clear executor cache according to the requested rerun mode
    if rerun is RerunMode.ALL:
        _clear_executor_cache(req_hash)
    elif rerun is RerunMode.FAILED:
        _clear_executor_cache(req_hash, failed_only=True)

    # Create new job record
    job_id = str(uuid4())

    job = Job(
        job_id=job_id,
        request_hash=req_hash,
        composition=norm_comp,
        potential=submission.potential,
        status="pending",
        request_data=submission.model_dump(),
        progress=_initial_progress(submission),
        tags=submission.tags or None,
    )
    store.create_job(job)
    logger.info("Created job %s (hash=%s)", job_id, req_hash)

    # Submit to executor
    try:
        resolved = _submit_to_executor(submission, job_id, req_hash)
        _update_from_resolved(job_id, resolved, submission)
    except Exception as exc:
        logger.exception("Failed to submit job %s", job_id)
        error_msg = f"Failed to start executor: {exc}"
        store.update_job(
            job_id,
            status="failed",
            errors={"submission": error_msg},
        )
        raise HTTPException(
            status_code=503,
            detail=(f"Job submission failed for job {job_id}. The compute backend is unavailable. {error_msg}"),
        ) from exc

    # Re-read to get final state
    final = store.get_job(job_id)
    errors = (final.errors or {}) if final else {}
    return JobCreatedResponse(
        id=job_id,
        status=JobStatus(final.status) if final else JobStatus.PENDING,
        composition=Composition.from_canonical(norm_comp),
        potential=submission.potential,
        tags=final.tags or [] if final else submission.tags,
        created_at=(final.created_at.isoformat() if final and final.created_at else _iso_now()),
        errors=errors,
        urls=_job_urls(job_id),
    )


@router.post(":search", response_model=JobSearchResponse, dependencies=[Depends(verify_token)])
def search_jobs(body: JobSearchRequest) -> JobSearchResponse:
    """Search for jobs across all lifecycle stages.

    Use this endpoint for job management: finding pending, running,
    completed, or failed jobs.  All filters are optional.  When
    *composition* is provided, only jobs with that exact composition
    are returned.  Use *statuses* to restrict to specific stages.

    To search completed results by composition similarity, use
    ``POST /glasses:search`` instead.
    """
    store = get_job_store()
    norm_comp = body.composition.canonical if body.composition else None

    statuses = [s.value for s in body.statuses] if body.statuses else None
    jobs = store.search_jobs(composition=norm_comp, potential=body.potential, statuses=statuses)
    matches = [
        JobSearchMatch(
            job_id=j.job_id,
            composition=Composition.from_canonical(j.composition),
            potential=j.potential,
            tags=j.tags or [],
            analyses=_analyses_list(j),
            status=JobStatus(j.status),
            created_at=j.created_at.isoformat() if j.created_at else None,
            completed_at=j.completed_at.isoformat() if j.completed_at else None,
            visualization_url=_job_urls(j.job_id)["visualization"],
        )
        for j in jobs
    ]

    # --- filter by tags (if requested) ---
    if body.tags:
        required = set(body.tags)
        matches = [m for m in matches if required <= set(m.tags)]

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
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        id=job.job_id,
        status=JobStatus(job.status),
        composition=Composition.from_canonical(job.composition),
        potential=job.potential,
        tags=job.tags or [],
        progress=_progress_from_dict(job.progress),
        errors=job.errors or {},
        created_at=job.created_at.isoformat() if job.created_at else _iso_now(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        urls=_job_urls(job.job_id),
    )


@router.post("/{job_id}:cancel", dependencies=[Depends(verify_token)])
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
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        id=job.job_id,
        status=JobStatus(job.status),
        composition=Composition.from_canonical(job.composition),
        potential=job.potential,
        tags=job.tags or [],
        progress=_progress_from_dict(job.progress),
        errors=job.errors or {},
        created_at=job.created_at.isoformat() if job.created_at else _iso_now(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        urls=_job_urls(job.job_id),
    )


@router.put("/{job_id}/tags", response_model=TagsResponse, dependencies=[Depends(verify_token)])
def update_tags(job_id: str, body: TagsUpdate) -> TagsResponse:
    """Replace the tags on a job."""
    store = get_job_store()
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    store.update_job(job_id, tags=sorted(set(body.tags)))
    return TagsResponse(job_id=job_id, tags=sorted(set(body.tags)))


@router.get("/{job_id}/settings", response_model=JobSettingsResponse)
def get_job_settings(job_id: str) -> JobSettingsResponse:
    """Return the original submission settings for a job."""
    store = get_job_store()
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.request_data:
        raise HTTPException(status_code=404, detail="No settings stored for this job")

    return JobSettingsResponse(
        job_id=job.job_id,
        settings=job.request_data,
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
        visualization_url=_job_urls(job.job_id)["visualization"],
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
    ] = "extxyz",
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


@router.get("/{job_id}/trajectory")
def get_trajectory(job_id: str) -> Response:
    """Return the full simulation history (trajectory) for a job.

    This is a large payload (can be hundreds of MB) containing per-stage
    positions, forces, velocities, and thermodynamic data.
    """
    store = get_job_store()

    # Fetch the raw JSON text directly from SQLite, bypassing the expensive
    # json.loads() → Python objects → json.dumps() round-trip.
    raw_history = store.get_raw_simulation_history(job_id)

    if raw_history is None:
        # Distinguish "job not found" from "no trajectory"
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        raise HTTPException(status_code=404, detail="No trajectory data available")

    # Fix NaN/Infinity tokens in the raw JSON string
    raw_history = raw_history.replace("NaN", "null").replace("Infinity", "null")

    raw = '{"job_id": "' + job_id + '", "simulation_history": ' + raw_history + "}"
    return Response(content=raw.encode("utf-8"), media_type="application/json")


def _render_error_html(job_id: str, title: str, detail: str, status_code: int = 400) -> HTMLResponse:
    """Render a user-facing HTML error page for the visualization endpoint."""
    from fastapi.templating import Jinja2Templates

    template_dir = Path(__file__).parent.parent / "templates"
    templates = Jinja2Templates(directory=str(template_dir))
    html = templates.get_template("error.html").render(
        job_id=job_id,
        title=title,
        detail=detail,
    )
    return HTMLResponse(content=html, status_code=status_code)


@router.get("/{job_id}/visualize", response_class=HTMLResponse)
def visualize_job(job_id: str) -> HTMLResponse:
    """Interactive HTML visualization of completed results."""
    from fastapi.templating import Jinja2Templates

    store = get_job_store()
    job = store.get_job(job_id)

    # Refresh running jobs so we pick up newly-completed steps
    if job and job.status == "running":
        refresh_job_from_cache(job)
        job = store.get_job(job_id)

    if not job:
        return _render_error_html(job_id, "Job Not Found", "No job exists with this ID.", 404)

    if job.status == "pending":
        return _render_error_html(job_id, "Job Pending", "This job has not started yet. Please check back later.", 400)

    if job.status == "failed" and not job.result_data:
        errors = job.errors or {}
        detail = "; ".join(f"{k}: {v}" for k, v in errors.items()) or "Unknown error"
        return _render_error_html(job_id, "Job Failed", detail, 422)

    result_data = job.result_data or {}

    if not result_data:
        return _render_error_html(job_id, "No Results Yet", "Results are not available yet for this job.", 404)

    try:
        context = build_visualization_context(
            job_id,
            result_data,
            request_hash=job.request_hash,
            request_data=job.request_data,
        )
        context["job_status"] = job.status
        context["progress"] = job.progress or {}
        context["tags"] = job.tags or []

        template_dir = Path(__file__).parent.parent / "templates"
        templates = Jinja2Templates(directory=str(template_dir))
        html_content = templates.get_template("results.html").render(context)

        return HTMLResponse(content=html_content)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating visualisation for job %s", job_id)
        return _render_error_html(job_id, "Visualisation Error", f"Error generating visualisation: {e!s}", 500)
