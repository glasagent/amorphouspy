"""Helpers for the jobs router — hashing, progress tracking, executor interaction."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from amorphouspy_api.config import MELTQUENCH_PROJECT_DIR
from amorphouspy_api.database import get_job_store
from amorphouspy_api.executor import get_executor, get_future_from_cache
from amorphouspy_api.models import (
    JobProgress,
    JobSubmission,
    StepStatus,
)
from amorphouspy_api.workflows import ANALYSES, BASE_STEPS, submit_pipeline

if TYPE_CHECKING:
    from amorphouspy_api.database import Job

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers
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
    """Convert a flat progress dict (from DB) into a JobProgress model."""
    if d is None:
        return JobProgress()
    base: dict[str, StepStatus] = {}
    analyses: dict[str, StepStatus] = {}
    for k, v in d.items():
        sv = StepStatus(v) if v is not None else StepStatus.PENDING
        if k in BASE_STEPS:
            base[k] = sv
        else:
            analyses[k] = sv
    return JobProgress(**base, analyses=analyses)


def _analyses_list(job: Job) -> list[str]:
    """Extract the list of requested analysis type names from a stored job."""
    req = job.request_data or {}
    return [a.get("type", "structure") for a in req.get("analyses", [])]


def _initial_progress(submission: JobSubmission) -> dict[str, str]:
    """Build the initial progress dict based on requested analyses."""
    progress: dict[str, str] = {
        "structure_generation": "pending",
        "melt_quench": "pending",
    }
    for a in submission.analyses:
        progress[a.type] = "pending"
    return progress


# ---------------------------------------------------------------------------
# Executor interaction
# ---------------------------------------------------------------------------


def _resolve_future(future, job_id: str) -> dict:
    """Extract state from an executorlib future."""
    if not future.done():
        return {"status": "running"}
    exc = future.exception()
    if exc is not None:
        logger.error("Job %s failed: %s", job_id, exc)
        return {"status": "failed", "error": str(exc)}
    return {"status": "completed", "result": future.result()}


def _submit_to_executor(
    submission: JobSubmission,
    job_id: str,
    request_hash: str,
) -> dict:
    """Submit the full pipeline to executorlib and return resolved status dict."""
    exe = get_executor(cache_directory=MELTQUENCH_PROJECT_DIR)
    future = submit_pipeline(exe, submission, cache_key=request_hash)
    resolved = _resolve_future(future, job_id)
    exe.shutdown(wait=False, cancel_futures=False)
    return resolved


def _update_from_resolved(job_id: str, resolved: dict, submission: JobSubmission | None = None) -> None:
    """Persist pipeline outcome into the job store."""
    store = get_job_store()
    status = resolved["status"]

    # Build the expected step list
    all_steps = list(BASE_STEPS)
    if submission:
        all_steps.extend(a.type for a in submission.analyses if a.type in ANALYSES)

    if status == "completed":
        result = resolved["result"]
        progress = dict.fromkeys(all_steps, "completed")
        store.update_job(
            job_id,
            status="completed",
            progress=progress,
            result_data=result,
            completed_at=datetime.now(UTC),
        )
    elif status == "failed":
        progress = dict.fromkeys(all_steps, "failed")
        store.update_job(
            job_id,
            status="failed",
            progress=progress,
            errors={"pipeline": resolved.get("error", "unknown error")},
        )
    else:
        progress = dict.fromkeys(all_steps, "pending")
        progress["structure_generation"] = "running"
        store.update_job(job_id, status="running", progress=progress)


def refresh_job_from_cache(job: Job) -> None:
    """Re-check executor cache for a running job and update DB if resolved."""
    try:
        future = get_future_from_cache(
            cache_directory=str(MELTQUENCH_PROJECT_DIR),
            cache_key=job.request_hash,
        )
        resolved = _resolve_future(future, job.job_id)
        submission = JobSubmission(**job.request_data) if job.request_data else None
        _update_from_resolved(job.job_id, resolved, submission)
    except FileNotFoundError:
        logger.info("Cache files not yet available for job %s", job.job_id)
    except Exception:
        logger.exception("Failed to check job %s", job.job_id)
