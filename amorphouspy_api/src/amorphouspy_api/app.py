"""amorphouspy Simulation API.

This module provides a FastAPI server for managing long-running glass simulation tasks.
It supports meltquench simulations for multi-component oxide glasses using the PMMCS
interatomic potential from Pedone et al.

Supported simulation types:
    - Meltquench simulations: Complete heating/cooling cycles for glass formation

Supported elements (PMMCS potential):
    Ag, Al, Ba, Be, Ca, Co, Cr, Cu, Er, Fe, Fe3, Gd, Ge, K, Li, Mg, Mn, Na, Nd, Ni, O, P, Sc, Si, Sn, Sr, Ti, Zn, Zr

Example usage:
    1. Start meltquench: POST /submit_meltquench -> returns task_id
    2. Check status: GET /check/{task_id} -> returns current status or results
"""

import hashlib
import logging
import os
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import cloudpickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi_mcp import FastApiMCP

from .database import get_task_store, init_task_store
from .jobs import JobManager
from .models import MeltquenchRequest, MeltquenchResult
from .visualization import router as visualization_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("glass_api.log")],
)
logger = logging.getLogger(__name__)

# Get amorphouspy version for project directory naming
try:
    amorphouspy_version = version("amorphouspy")
    logger.info("Using amorphouspy version: %s", amorphouspy_version)
except Exception:
    amorphouspy_version = "unknown"
    logger.warning("Could not determine amorphouspy version, using 'unknown'")

# Setup shared project directory
PROJECTS_FOLDER = Path(__file__).resolve().parent.parent.parent / "projects"

# Check for AMORPHOUSPY_PROJECTS environment variable
if "AMORPHOUSPY_PROJECTS" in os.environ:
    PROJECTS_FOLDER = Path(os.environ["AMORPHOUSPY_PROJECTS"])
    logger.info("Using project directory from AMORPHOUSPY_PROJECTS: %s", PROJECTS_FOLDER)
else:
    logger.info("Using default project directory: %s", PROJECTS_FOLDER)

MELTQUENCH_PROJECT_DIR = PROJECTS_FOLDER / f"amorphouspy_{amorphouspy_version}" / "meltquench"


# Configure API base URL for visualization links
API_BASE_URL = os.environ.get("API_BASE_URL", "")
if API_BASE_URL:
    logger.info("Using API base URL for visualization links: %s", API_BASE_URL)
else:
    logger.info("No API base URL configured, using relative paths")

# Ensure the projects directory exists
PROJECTS_FOLDER.mkdir(parents=True, exist_ok=True)
logger.info("Ensured projects directory exists: %s", PROJECTS_FOLDER)

# Initialize persistent task store
DB_PATH = PROJECTS_FOLDER / "tasks.db"
logger.info("Task store database path: %s", DB_PATH)
logger.info(
    "Directory exists: %s, Directory writable: %s",
    PROJECTS_FOLDER.exists(),
    os.access(PROJECTS_FOLDER, os.W_OK) if PROJECTS_FOLDER.exists() else "N/A",
)
init_task_store(DB_PATH)
_task_store = get_task_store()

# Initialize job manager (executor type configured via EXECUTOR_TYPE env var)
_job_manager = JobManager(cache_directory=MELTQUENCH_PROJECT_DIR)


def get_meltquench_hash(request: MeltquenchRequest) -> str:
    """Compute hash for a meltquench request to enable caching.

    Args:
        request: The meltquench request object to hash.

    Returns:
        First 16 characters of the SHA256 hash of the request parameters.
    """
    # Create sorted component-value pairs for consistent hashing
    comp_value_pairs = sorted(zip(request.components, request.values, strict=True))

    hash_params = {
        "composition": comp_value_pairs,
        "unit": request.unit,
        "heating_rate": request.heating_rate,
        "cooling_rate": request.cooling_rate,
        "n_print": request.n_print,
        "n_atoms": request.n_atoms,
    }

    # Use cloudpickle for consistent serialization, then hash with sha256
    binary_data = cloudpickle.dumps(hash_params)
    return hashlib.sha256(binary_data).hexdigest()[:16]  # First 16 chars for brevity


def get_visualization_url(task_id: str) -> str:
    """Construct the full visualization URL for a given task ID.

    Args:
        task_id: The unique identifier for the task.

    Returns:
        The full URL or relative path to the visualization page.
    """
    relative_path = f"/visualize/meltquench/{task_id}"
    if API_BASE_URL:
        # Remove trailing slash from base URL if present, then combine
        base_url = API_BASE_URL.rstrip("/")
        return f"{base_url}{relative_path}"
    return relative_path


# Create FastAPI app
app = FastAPI(
    title="amorphouspy Simulation API",
    description="API for managing long-running glass simulation tasks using amorphouspy",
    version="0.1.0",
)

# Enable CORS for all origins (customize as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include visualization router
app.include_router(visualization_router, tags=["visualization"])


@app.post("/cache/meltquench", tags=["tool"])
async def check_cached_result(request: MeltquenchRequest) -> MeltquenchResult | None:
    """Check if a result for the given meltquench request is already available in cache.

    Args:
        request: The meltquench request to check.

    Returns:
        The cached result if found, otherwise None.

    Raises:
        HTTPException: If an error occurs during the check.
    """
    try:
        request_hash = get_meltquench_hash(request)
        logger.info("Checking for cached result with hash: %s", request_hash)

        # Use database's efficient hash-based lookup
        cached_result = _task_store.find_cached_result(request_hash)

        if cached_result:
            logger.info("Found cached result")
            # Return just the result, not the task_id (for API compatibility)
            return cached_result[1]

        logger.info("No cached result found")
        return None

    except Exception:
        logger.exception("Error checking cached result")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@app.post("/submit/meltquench", tags=["tool"])
async def submit_meltquench(request: MeltquenchRequest) -> dict:
    """Start a new meltquench simulation task.

    This endpoint submits a meltquench job using executorlib.
    If the job with identical parameters has already been submitted,
    it will return the cached result or current status.

    Note: Results can be visualized at /visualize/meltquench/{task_id}

    Args:
        request: The meltquench request parameters.

    Returns:
        A dictionary containing the task ID, status, and visualization URL.

    Raises:
        HTTPException: If the task cannot be started.
    """
    try:
        request_hash = get_meltquench_hash(request)
        request_data = request.model_dump()

        # Check if we already have a cached result in our database
        cached_result = _task_store.find_cached_result(request_hash)
        if cached_result:
            cached_task_id, cached_meltquench_result = cached_result
            logger.info("Returning cached result from task %s", cached_task_id)
            return {
                "task_id": cached_task_id,
                "status": "completed_from_cache",
                "visualization_url": get_visualization_url(cached_task_id),
                "result": cached_meltquench_result.model_dump(),
            }

        task_id = str(uuid4())
        logger.info("Submitting meltquench task with ID: %s, hash: %s", task_id, request_hash)

        # Submit job via executorlib
        # This will either start a new job or return cached status
        job_status = _job_manager.submit_meltquench(request_data=request_data)

        # Store task in database
        _task_store.set(
            task_id,
            {
                "state": job_status["state"],
                "status": job_status["status"],
                "request_hash": request_hash,
                "request_data": request.model_dump(),
                "result": job_status.get("result"),
                "error": job_status.get("error"),
            },
        )

        if job_status["state"] == "complete":
            return {
                "task_id": task_id,
                "status": "completed",
                "visualization_url": get_visualization_url(task_id),
                "result": job_status["result"],
            }

        return {
            "task_id": task_id,
            "status": job_status["status"],
            "visualization_url": get_visualization_url(task_id),
        }
    except Exception:
        logger.exception("Error submitting meltquench task")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@app.get("/check/{task_id}", tags=["tool"])
async def check(task_id: str) -> dict:
    """Check the current status of a simulation task by its ID.

    This endpoint re-submits the job parameters to check status.
    If the job is complete, the cached result is returned.
    If still running, the current status is returned.

    Note: When ready, visualize results at /visualize/meltquench/{task_id}

    Args:
        task_id: The ID of the task to check.

    Returns:
        A dictionary containing the task status, result (if available), and visualization URL.

    Raises:
        HTTPException: If the task is not found.
    """
    meta = _task_store.get(task_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Task not found")

    # If already complete or errored in our database, return that
    if meta["state"] in ("complete", "error"):
        return {
            "task_id": task_id,
            "state": meta["state"],
            "status": meta.get("status", "processing"),
            "visualization_url": get_visualization_url(task_id),
            "error": meta.get("error"),
            "result": meta.get("result"),
        }

    # For running jobs, re-check by re-submitting
    # executorlib's caching will return the running future or cached result
    request_data = meta.get("request_data")
    if request_data:
        job_status = _job_manager.check_status(request_data=request_data)

        # Update database if status changed
        if job_status["state"] != meta["state"]:
            meta.update(
                {
                    "state": job_status["state"],
                    "status": job_status["status"],
                    "result": job_status.get("result"),
                    "error": job_status.get("error"),
                }
            )
            _task_store.set(task_id, meta)

        return {
            "task_id": task_id,
            "state": job_status["state"],
            "status": job_status["status"],
            "visualization_url": get_visualization_url(task_id),
            "error": job_status.get("error"),
            "result": job_status.get("result"),
        }

    # Fallback to database state
    return {
        "task_id": task_id,
        "state": meta["state"],
        "status": meta.get("status", "processing"),
        "visualization_url": get_visualization_url(task_id),
        "error": meta.get("error"),
        "result": meta.get("result"),
    }


mcp = FastApiMCP(app, include_tags=["tool"])
mcp.mount_http(mount_path="/mcp")


@app.get("/")
async def root() -> RedirectResponse:
    """Root endpoint redirects to API documentation."""
    return RedirectResponse(url="/docs")
