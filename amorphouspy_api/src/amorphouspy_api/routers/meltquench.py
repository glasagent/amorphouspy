"""Meltquench simulation router.

Endpoints for submitting, checking, and caching meltquench simulations.
"""

import hashlib
import logging
from uuid import uuid4

import cloudpickle
from executorlib import get_future_from_cache
from fastapi import APIRouter, HTTPException

from amorphouspy_api.config import API_BASE_URL, MELTQUENCH_PROJECT_DIR
from amorphouspy_api.database import get_task_store
from amorphouspy_api.jobs import get_executor, get_lammps_resource_dict
from amorphouspy_api.models import (
    MeltquenchRequest,
    MeltquenchResult,
    TaskResponse,
    TaskStatus,
)
from amorphouspy_api.workflows import run_meltquench_workflow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

router = APIRouter()


def get_meltquench_hash(request: MeltquenchRequest) -> str:
    """Compute hash for a meltquench request to enable caching.

    Args:
        request: The meltquench request object to hash.

    Returns:
        First 16 characters of the SHA256 hash of the request parameters.
    """
    comp_value_pairs = sorted(zip(request.components, request.values, strict=True))

    hash_params = {
        "composition": comp_value_pairs,
        "unit": request.unit,
        "heating_rate": request.heating_rate,
        "cooling_rate": request.cooling_rate,
        "n_print": request.n_print,
        "n_atoms": request.n_atoms,
    }

    binary_data = cloudpickle.dumps(hash_params)
    return hashlib.sha256(binary_data).hexdigest()[:16]


def get_visualization_url(task_id: str) -> str:
    """Construct the full visualization URL for a given task ID.

    Args:
        task_id: The unique identifier for the task.

    Returns:
        The full URL or relative path to the visualization page.
    """
    relative_path = f"/visualize/meltquench/{task_id}"
    if API_BASE_URL:
        base_url = API_BASE_URL.rstrip("/")
        return f"{base_url}{relative_path}"
    return relative_path


def resolve_future(future, task_id: str) -> dict:
    """Extract state, result, and error from a resolved or pending future.

    Args:
        future: A concurrent.futures.Future-like object.
        task_id: The task identifier (used for logging).

    Returns:
        A dict with 'state' and optionally 'result' or 'error' keys.
    """
    if not future.done():
        return {"state": "running"}

    exc = future.exception()
    if exc is not None:
        error_msg = str(exc)
        logger.error("Task %s failed: %s", task_id, error_msg)
        return {"state": "error", "error": error_msg}

    serialized = MeltquenchResult(**future.result()).model_dump()
    return {"state": "complete", "result": serialized}


def build_task_response(
    task_id: str,
    job_status: dict,
    *,
    from_cache: bool = False,
) -> TaskResponse:
    """Build a TaskResponse from job status.

    Args:
        task_id: The task identifier.
        job_status: Dictionary with 'state', 'result', and 'error' keys.
        from_cache: Whether this result was retrieved from cache.

    Returns:
        A TaskResponse model instance.
    """
    state = job_status["state"]

    if state == "complete":
        status = TaskStatus.COMPLETED_FROM_CACHE if from_cache else TaskStatus.COMPLETED
        result = MeltquenchResult(**job_status["result"]) if job_status.get("result") else None
    elif state == "error":
        status = TaskStatus.ERROR
        result = None
    else:  # running
        status = TaskStatus.RUNNING
        result = None

    return TaskResponse(
        task_id=task_id,
        status=status,
        visualization_url=get_visualization_url(task_id),
        result=result,
        error=job_status.get("error"),
    )


def submit_to_executor(
    request_data: dict,
    task_id: str,
    request_hash: str,
    *,
    cache_key: str | None = None,
) -> dict:
    """Submit a meltquench job to the executor and resolve its status.

    The executor's disk cache (``MELTQUENCH_PROJECT_DIR``) means that a
    previously-completed job will have ``done() == True`` immediately.

    Args:
        request_data: Dictionary with the meltquench request parameters.
        task_id: The unique task identifier.
        request_hash: Hash of the request for caching.
        cache_key: Optional explicit cache key for the final workflow step,
            enabling later retrieval via ``get_future_from_cache``.

    Returns:
        A job-status dict with 'state', 'result', and 'error' keys.
    """
    exe = get_executor(cache_directory=MELTQUENCH_PROJECT_DIR)
    lammps_resource_dict = get_lammps_resource_dict()
    future = run_meltquench_workflow(
        executor=exe,
        components=request_data["components"],
        values=request_data["values"],
        n_atoms=request_data["n_atoms"],
        potential_type=request_data["potential_type"],
        heating_rate=request_data["heating_rate"],
        cooling_rate=request_data["cooling_rate"],
        n_print=request_data["n_print"],
        lammps_resource_dict=lammps_resource_dict,
        cache_key=cache_key,
    )

    # Resolve the future while the executor is still active
    task_store = get_task_store()

    meta = {
        "request_hash": request_hash,
        "request_data": request_data,
        **resolve_future(future, task_id),
    }

    task_store.set(task_id, meta)
    exe.shutdown(wait=False, cancel_futures=False)

    # Note: after shutdown of executor, do not touch the future anymore
    # E.g. the FluxClusterExecutor will cancel the Future object (while not cancelling the underlying job)
    # See https://github.com/pyiron/executorlib/issues/921#issuecomment-3919953044

    return meta


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/cache/meltquench", tags=["tool"])
def check_cached_result(request: MeltquenchRequest) -> MeltquenchResult | None:
    """Check if a result for the given meltquench request is already available in cache.

    Args:
        request: The meltquench request to check.

    Returns:
        The cached result if found, otherwise None.

    Raises:
        HTTPException: If an error occurs during the check.
    """
    try:
        task_store = get_task_store()
        request_hash = get_meltquench_hash(request)
        logger.info("Checking for cached result with hash: %s", request_hash)

        cached_result = task_store.find_cached_result(request_hash)

        if cached_result:
            logger.info("Found cached result")
            return cached_result[1]

        logger.info("No cached result found")
        return None

    except Exception:
        logger.exception("Error checking cached result")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.post("/submit/meltquench", tags=["tool"])
def submit_meltquench(request: MeltquenchRequest) -> TaskResponse:
    """Start a new meltquench simulation task.

    Submit a melt-quench simulation for multi-component oxide glasses.
    The calculation uses the PMMCS interatomic potential (Pedone et al.)
    and runs a complete heating / cooling cycle for glass formation.

    Supported elements (PMMCS potential):
        Ag, Al, Ba, Be, Ca, Co, Cr, Cu, Er, Fe, Fe3, Gd, Ge, K, Li,
        Mg, Mn, Na, Nd, Ni, O, P, Sc, Si, Sn, Sr, Ti, Zn, Zr

    If the job with identical parameters has already been submitted,
    it will return the cached result or current status.

    Note: Results can be visualized at /visualize/meltquench/{task_id}

    Args:
        request: The meltquench request parameters.

    Returns:
        TaskResponse with task ID, status, and result if available.

    Raises:
        HTTPException: If the task cannot be started.
    """
    try:
        task_store = get_task_store()
        request_hash = get_meltquench_hash(request)
        request_data = request.model_dump()

        # Check if we already have a cached result in our database
        cached_result = task_store.find_cached_result(request_hash)
        if cached_result:
            cached_task_id, cached_meltquench_result = cached_result
            logger.info("Returning cached result from task %s", cached_task_id)
            return build_task_response(
                cached_task_id,
                {"state": "complete", "result": cached_meltquench_result.model_dump()},
                from_cache=True,
            )

        task_id = str(uuid4())
        logger.info("Submitting meltquench task with ID: %s, hash: %s", task_id, request_hash)
        status = submit_to_executor(request_data, task_id, request_hash, cache_key=request_hash)
        return build_task_response(task_id, status)

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error submitting meltquench task")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/check/{task_id}", tags=["tool"])
def check(task_id: str) -> TaskResponse:
    """Check the current status of a simulation task by its ID.

    Uses ``get_future_from_cache()`` to recreate the future from the
    executor's disk cache, avoiding re-submission of the entire workflow.

    Note: When ready, visualize results at /visualize/meltquench/{task_id}

    Args:
        task_id: The ID of the task to check.

    Returns:
        TaskResponse with current status, result (if available), and visualization URL.

    Raises:
        HTTPException: If the task is not found.
    """
    task_store = get_task_store()
    meta = task_store.get(task_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Task not found")
    logger.info("check %s: state=%s", task_id, meta["state"])

    if meta["state"] != "running":
        return build_task_response(task_id, meta)

    request_hash = meta.get("request_hash", "")
    request_data = meta.get("request_data", {})

    if not request_hash:
        raise HTTPException(status_code=500, detail="Task is missing request hash")

    # Recreate the future from the executor's disk cache instead of
    # re-submitting the entire workflow.  See
    # https://github.com/pyiron/executorlib/pull/915
    try:
        future = get_future_from_cache(
            cache_directory=str(MELTQUENCH_PROJECT_DIR),
            cache_key=request_hash,
        )

        status = {
            "request_hash": request_hash,
            "request_data": request_data,
            **resolve_future(future, task_id),
        }

        task_store.set(task_id, status)
    except FileNotFoundError:
        # Cache files not yet written - job is still starting up
        logger.info("Cache files not yet available for task %s", task_id)
        status = {"state": "running", "request_hash": request_hash, "request_data": request_data}
    except Exception as exc:
        logger.exception("Failed to check task %s", task_id)
        error_msg = str(exc)
        status = {"state": "error", "error": error_msg}
        task_store.set(
            task_id,
            {"state": "error", "request_hash": request_hash, "request_data": request_data, "error": error_msg},
        )
    return build_task_response(task_id, status)
