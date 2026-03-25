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

import hashlib
import json
import logging
from datetime import UTC, datetime
from io import StringIO
from typing import Annotated
from uuid import uuid4

from ase.io import write as ase_write
from executorlib import get_future_from_cache
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

from amorphouspy_api.composition import Composition
from amorphouspy_api.config import MELTQUENCH_PROJECT_DIR
from amorphouspy_api.database import Job, get_job_store
from amorphouspy_api.jobs import get_executor, get_lammps_resource_dict
from amorphouspy_api.models import (
    JobCreatedResponse,
    JobProgress,
    JobResultsResponse,
    JobSearchMatch,
    JobSearchRequest,
    JobSearchResponse,
    JobStatus,
    JobStatusResponse,
    JobSubmission,
    StepStatus,
    ViscosityAnalysis,
    validate_atoms,
)
from amorphouspy_api.workflows import run_meltquench_workflow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["tool"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _job_hash(submission: JobSubmission, normalized_comp: str) -> str:
    """Deterministic hash from (normalised composition + potential + sim params + analyses)."""
    analyses_dump = [a.model_dump() for a in submission.analyses]
    payload = json.dumps(
        {
            "composition": normalized_comp,
            "potential": submission.potential.value,
            "simulation": submission.simulation.model_dump(),
            "analyses": analyses_dump,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _progress_from_dict(d: dict | None) -> JobProgress:
    if d is None:
        return JobProgress()
    fields = {}
    for k, v in d.items():
        if v is None:
            fields[k] = None
        else:
            fields[k] = StepStatus(v)
    return JobProgress(**fields)


def _analyses_list(job: Job) -> list[str]:
    """Extract the list of requested analysis type names from a stored job."""
    req = job.request_data or {}
    return [a.get("type", "structure") for a in req.get("analyses", [])]


def _resolve_future(future, job_id: str) -> dict:
    """Extract state from an executorlib future."""
    if not future.done():
        return {"status": "running"}
    exc = future.exception()
    if exc is not None:
        logger.error("Job %s failed: %s", job_id, exc)
        return {"status": "failed", "error": str(exc)}
    return {"status": "completed", "result": future.result()}


def _wants_viscosity(submission: JobSubmission) -> ViscosityAnalysis | None:
    """Return the ViscosityAnalysis config if the submission requests it."""
    for a in submission.analyses:
        if isinstance(a, ViscosityAnalysis):
            return a
    return None


def _initial_progress(submission: JobSubmission) -> dict[str, str]:
    """Build the initial progress dict based on requested analyses."""
    progress = {
        "structure_generation": "pending",
        "melt_quench": "pending",
        "structure_analysis": "pending",
    }
    if _wants_viscosity(submission):
        progress["viscosity"] = "pending"
    return progress


def _submit_to_executor(
    submission: JobSubmission,
    job_id: str,
    request_hash: str,
    composition: dict[str, float],
) -> dict:
    """Submit workflow to executorlib and return resolved status dict."""
    exe = get_executor(cache_directory=MELTQUENCH_PROJECT_DIR)
    lammps_resource_dict = get_lammps_resource_dict()

    future = run_meltquench_workflow(
        executor=exe,
        composition=composition,
        n_atoms=submission.simulation.n_atoms,
        potential_type=submission.potential.value,
        heating_rate=int(submission.simulation.quench_rate * 100),  # default heating = 100x quench
        cooling_rate=int(submission.simulation.quench_rate),
        n_print=1000,
        lammps_resource_dict=lammps_resource_dict,
        cache_key=request_hash,
    )

    resolved = _resolve_future(future, job_id)
    exe.shutdown(wait=False, cancel_futures=False)
    return resolved


def _update_job_from_resolved(job_id: str, resolved: dict, submission: JobSubmission | None = None) -> None:
    """Persist executor outcome into the job store.

    If the melt-quench completed and viscosity was requested, chain the
    viscosity workflow before marking the job as completed.
    """
    store = get_job_store()
    status = resolved["status"]

    visc_cfg = _wants_viscosity(submission) if submission else None

    if status == "completed":
        result = resolved["result"]
        if visc_cfg is not None:
            # Mark melt-quench done, viscosity running
            store.update_job(
                job_id,
                status="running",
                progress={
                    "structure_generation": "completed",
                    "melt_quench": "completed",
                    "structure_analysis": "completed",
                    "viscosity": "running",
                },
                result_data=result,
            )
            try:
                visc_result = _run_viscosity_chain(
                    submission=submission,
                    visc_cfg=visc_cfg,
                )
                # Merge viscosity result into existing result_data
                result["viscosity"] = visc_result
                store.update_job(
                    job_id,
                    status="completed",
                    progress={
                        "structure_generation": "completed",
                        "melt_quench": "completed",
                        "structure_analysis": "completed",
                        "viscosity": "completed",
                    },
                    result_data=result,
                    completed_at=datetime.now(UTC),
                )
            except Exception as exc:
                logger.exception("Viscosity workflow failed for job %s", job_id)
                store.update_job(
                    job_id,
                    status="failed",
                    progress={
                        "structure_generation": "completed",
                        "melt_quench": "completed",
                        "structure_analysis": "completed",
                        "viscosity": "failed",
                    },
                    errors={"viscosity": str(exc)},
                )
        else:
            store.update_job(
                job_id,
                status="completed",
                progress={
                    "structure_generation": "completed",
                    "melt_quench": "completed",
                    "structure_analysis": "completed",
                },
                result_data=result,
                completed_at=datetime.now(UTC),
            )
    elif status == "failed":
        store.update_job(
            job_id,
            status="failed",
            progress={
                "structure_generation": "completed",
                "melt_quench": "failed",
                "structure_analysis": "pending",
            },
            errors={"melt_quench": resolved.get("error", "unknown error")},
        )
    else:
        progress = {
            "structure_generation": "running",
            "melt_quench": "pending",
            "structure_analysis": "pending",
        }
        if visc_cfg:
            progress["viscosity"] = "pending"
        store.update_job(job_id, status="running", progress=progress)


def _run_viscosity_chain(
    submission: JobSubmission,
    visc_cfg: ViscosityAnalysis,
) -> dict:
    """Run the viscosity workflow after melt-quench."""
    from amorphouspy_api.workflows.viscosity import run_viscosity_workflow

    lammps_resource_dict = get_lammps_resource_dict()
    return run_viscosity_workflow(
        composition=submission.composition.root,
        n_atoms=submission.simulation.n_atoms,
        potential_type=submission.potential.value,
        heating_rate=int(submission.simulation.quench_rate * 100),
        cooling_rate=int(submission.simulation.quench_rate),
        temperatures=visc_cfg.temperatures,
        timestep=visc_cfg.timestep,
        n_timesteps=visc_cfg.n_timesteps,
        n_print=visc_cfg.n_print,
        max_lag=visc_cfg.max_lag,
        lammps_resource_dict=lammps_resource_dict,
    )


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
            created_at=cached.created_at.isoformat() if cached.created_at else _iso_now(),
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
        resolved = _submit_to_executor(submission, job_id, req_hash, submission.composition.root)
        _update_job_from_resolved(job_id, resolved, submission)
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
        created_at=final.created_at.isoformat() if final and final.created_at else _iso_now(),
    )


@router.post(":search", response_model=JobSearchResponse)
def search_jobs(request: JobSearchRequest) -> JobSearchResponse:
    """Search for existing completed / running jobs matching a spec."""
    store = get_job_store()
    norm_comp = request.composition.canonical
    jobs = store.search_by_composition(norm_comp, request.potential)

    matches = [
        JobSearchMatch(
            job_id=j.job_id,
            composition=Composition.from_canonical(j.composition),
            potential=j.potential,
            analyses=_analyses_list(j),
            similarity=1.0,
            completed_at=j.completed_at.isoformat() if j.completed_at else None,
        )
        for j in jobs
    ]
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
        req_hash = job.request_hash
        try:
            future = get_future_from_cache(
                cache_directory=str(MELTQUENCH_PROJECT_DIR),
                cache_key=req_hash,
            )
            resolved = _resolve_future(future, job_id)
            # Reconstruct submission for viscosity chaining
            submission = JobSubmission(**job.request_data) if job.request_data else None
            _update_job_from_resolved(job_id, resolved, submission)
            job = store.get_job(job_id)
        except FileNotFoundError:
            logger.info("Cache files not yet available for job %s", job_id)
        except Exception:
            logger.exception("Failed to check job %s", job_id)

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

    return JobResultsResponse(
        job_id=job.job_id,
        composition=Composition.from_canonical(job.composition),
        structure=result.get("structural_analysis"),
        viscosity=result.get("viscosity"),
    )


@router.get("/{job_id}/results/{analysis}")
def get_single_result(job_id: str, analysis: str) -> dict:
    """Results for a single analysis type."""
    store = get_job_store()
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = job.result_data or {}
    # Map analysis name to result key
    key_map = {
        "structure": "structural_analysis",
        "elastic": "elastic",
        "viscosity": "viscosity",
        "cte": "cte",
    }
    key = key_map.get(analysis, analysis)
    if key not in result:
        raise HTTPException(status_code=404, detail=f"No results for analysis '{analysis}'")

    return {"job_id": job.job_id, analysis: result[key]}


@router.get("/{job_id}/structure")
def get_structure(
    job_id: str,
    fmt: Annotated[str, Query(alias="format", description="Export format: xyz, cif, poscar, extxyz")] = "xyz",
) -> Response:
    """Export the final quenched structure."""
    store = get_job_store()
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = job.result_data or {}
    raw_structure = result.get("final_structure")
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
    from amorphouspy_api.visualization import render_job_visualization

    return render_job_visualization(job_id)
