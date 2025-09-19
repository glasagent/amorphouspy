"""Pyiron Glass Simulation API.

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

import asyncio
import concurrent.futures
import hashlib
import logging
import os
from pathlib import Path
from uuid import uuid4

import cloudpickle
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi_mcp import FastApiMCP

from .database import get_task_store, init_task_store
from .models import MeltquenchRequest, MeltquenchResult
from .visualization import router as visualization_router
from .worker import meltquench_worker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("glass_api.log")],
)
logger = logging.getLogger(__name__)

# Setup shared project directory using canonical pyiron environment variables
PROJECTS_FOLDER = Path(__file__).resolve().parent.parent.parent / "projects"

# Check for PYIRONPROJECTPATHS environment variables
if "PYIRONPROJECTPATHS" in os.environ:
    PROJECTS_FOLDER = Path(os.environ["PYIRONPROJECTPATHS"])
    logger.info("Using project directory from PYIRONPROJECTPATHS: %s", PROJECTS_FOLDER)
else:
    logger.info("Using default project directory: %s", PROJECTS_FOLDER)

MELTQUENCH_PROJECT_DIR = PROJECTS_FOLDER / "meltquench"

# Configure pyiron environment variables if not already set
if "PYIRONPROJECTPATHS" not in os.environ:
    os.environ["PYIRONPROJECTPATHS"] = str(PROJECTS_FOLDER)
    logger.info("Set PYIRONPROJECTPATHS to: %s", PROJECTS_FOLDER)

if "PYIRONSQLCONNECTIONSTRING" not in os.environ:
    pyiron_db_path = PROJECTS_FOLDER / "pyiron.db"
    os.environ["PYIRONSQLCONNECTIONSTRING"] = f"sqlite:///{pyiron_db_path}"
    logger.info("Set PYIRONSQLCONNECTIONSTRING to: sqlite:///%s", pyiron_db_path)

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


def get_meltquench_hash(request: MeltquenchRequest) -> str:
    """Compute hash for a meltquench request to enable caching."""
    # Create sorted component-value pairs for consistent hashing
    comp_value_pairs = sorted(zip(request.components, request.values, strict=True))

    hash_params = {
        "composition": comp_value_pairs,
        "unit": request.unit,
        "heating_rate": request.heating_rate,
        "cooling_rate": request.cooling_rate,
        "n_print": request.n_print,
    }

    # Use cloudpickle for consistent serialization, then hash with sha256
    binary_data = cloudpickle.dumps(hash_params)
    return hashlib.sha256(binary_data).hexdigest()[:16]  # First 16 chars for brevity


def get_visualization_url(task_id: str) -> str:
    """Construct the full visualization URL for a given task ID."""
    relative_path = f"/visualize/meltquench/{task_id}"
    if API_BASE_URL:
        # Remove trailing slash from base URL if present, then combine
        base_url = API_BASE_URL.rstrip("/")
        return f"{base_url}{relative_path}"
    return relative_path


async def _meltquench_worker(task_id: str, request: MeltquenchRequest) -> None:
    """Async wrapper for meltquench simulation that runs the synchronous worker in a process executor.

    Args:
        task_id (str): Unique identifier for the task
        request (MeltquenchRequest): Validated meltquench parameters

    """
    loop = asyncio.get_event_loop()

    # Convert request to dict for serialization across processes
    request_dict = request.model_dump()

    # Run the synchronous worker in a process executor to handle pyiron's signal handling
    with concurrent.futures.ProcessPoolExecutor() as executor:
        await loop.run_in_executor(
            executor, meltquench_worker, task_id, request_dict, DB_PATH, str(MELTQUENCH_PROJECT_DIR)
        )


# Create FastAPI app
app = FastAPI(
    title="Pyiron Glass Simulation API",
    description="API for managing long-running glass simulation tasks using pyiron-glass",
    version="0.1.0",
)

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include visualization router
app.include_router(visualization_router, tags=["visualization"])


@app.post("/cache/meltquench", tags=["tool"])
async def check_cached_result(request: MeltquenchRequest) -> MeltquenchResult | None:
    """Check if a result for the given meltquench request is already available in cache."""
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

    Note: Results can be visualized at /visualize/meltquench/{task_id}
    """
    try:
        # Check if we already have a cached result
        request_hash = get_meltquench_hash(request)
        cached_result = _task_store.find_cached_result(request_hash)

        if cached_result:
            cached_task_id, cached_meltquench_result = cached_result
            logger.info("Returning cached result from task %s instead of starting new task", cached_task_id)
            return {
                "task_id": cached_task_id,
                "status": "completed_from_cache",
                "visualization_url": get_visualization_url(cached_task_id),
                "result": cached_meltquench_result.model_dump(),
            }

        task_id = str(uuid4())
        logger.info("Creating new meltquench task with ID: %s, hash: %s", task_id, request_hash)

        # Store task in database
        _task_store.set(
            task_id,
            {
                "state": "processing",
                "status": "Initializing",
                "request_hash": request_hash,
                "request_data": request.model_dump(),  # Store original request for reference
            },
        )

        # Always run as background task using process executor
        task = asyncio.create_task(_meltquench_worker(task_id, request))
        # Store task reference to prevent garbage collection
        task.add_done_callback(lambda _: None)

        return {"task_id": task_id, "status": "started", "visualization_url": get_visualization_url(task_id)}
    except Exception:
        logger.exception("Error starting meltquench task")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@app.get("/check/{task_id}", tags=["tool"])
async def check(task_id: str) -> dict:
    """Check the current status of a simulation task by its ID.

    Note: When ready, visualize results at /visualize/meltquench/{task_id}
    """
    meta = _task_store.get(task_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Task not found")

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
