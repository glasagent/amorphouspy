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
from fastapi_mcp import FastApiMCP

from .database import get_task_store, init_task_store
from .models import MeltquenchRequest, MeltquenchResult
from .worker import meltquench_worker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("glass_api.log")],
)
logger = logging.getLogger(__name__)

# Setup shared project directory - check for environment variable first
SCRATCH_FOLDER = Path(__file__).resolve().parent.parent.parent / "scratch"
if "PYIRON_SCRATCH_FOLDER" in os.environ:
    SCRATCH_FOLDER = Path(os.environ["PYIRON_SCRATCH_FOLDER"])
    logger.info("Using custom scratch directory from PYIRON_SCRATCH_FOLDER: %s", SCRATCH_FOLDER)

SHARED_PROJECT_DIR = SCRATCH_FOLDER / "meltquench"

# Initialize persistent task store
DB_PATH = SCRATCH_FOLDER / "tasks.db"
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


async def _meltquench_worker(task_id: str, request: MeltquenchRequest) -> None:
    """Async wrapper for meltquench simulation that runs the synchronous worker in a process executor.

    Args:
        task_id (str): Unique identifier for the task
        request (MeltquenchRequest): Validated meltquench parameters

    """
    loop = asyncio.get_event_loop()

    # Convert request to dict for serialization across processes
    request_dict = request.model_dump()

    # Use ThreadPoolExecutor for testing to get better coverage
    # Use ProcessPoolExecutor in production to handle pyiron's signal handling
    use_threads = os.environ.get("PYIRON_GLASS_USE_THREADS", "false").lower() == "true"

    if use_threads:
        logger.info("Using ThreadPoolExecutor for testing/coverage")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor, meltquench_worker, task_id, request_dict, DB_PATH, str(SHARED_PROJECT_DIR)
            )
    else:
        # Run the synchronous worker in a process executor to handle pyiron's signal handling
        with concurrent.futures.ProcessPoolExecutor() as executor:
            await loop.run_in_executor(
                executor, meltquench_worker, task_id, request_dict, DB_PATH, str(SHARED_PROJECT_DIR)
            )


# Create FastAPI app
app = FastAPI(
    title="Pyiron Glass Simulation API",
    description="API for managing long-running glass simulation tasks using pyiron-glass",
    version="0.1.0",
)


@app.post("/check_cached_result", tags=["tool"])
async def check_cached_result(request: MeltquenchRequest) -> MeltquenchResult | None:
    """Check if a result for the given meltquench request is already available in cache."""
    try:
        request_hash = get_meltquench_hash(request)
        logger.info("Checking for cached result with hash: %s", request_hash)

        # Use database's efficient hash-based lookup
        cached_result = _task_store.find_cached_result(request_hash)

        if cached_result:
            logger.info("Found cached result")
            return cached_result

        logger.info("No cached result found")
        return None

    except Exception:
        logger.exception("Error checking cached result")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@app.post("/submit_meltquench", tags=["tool"])
async def submit_meltquench(request: MeltquenchRequest) -> dict:
    """Start a new meltquench simulation task."""
    try:
        # Check if we already have a cached result
        cached_result = await check_cached_result(request)
        if cached_result:
            logger.info("Returning cached result instead of starting new task")
            return {"task_id": "cached", "status": "completed_from_cache", "result": cached_result.model_dump()}

        task_id = str(uuid4())
        request_hash = get_meltquench_hash(request)
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

        return {"task_id": task_id, "status": "started"}
    except Exception:
        logger.exception("Error starting meltquench task")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@app.get("/check/{task_id}", tags=["tool"])
async def check(task_id: str) -> dict:
    """Check the current status of a simulation task by its ID."""
    meta = _task_store.get(task_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task_id,
        "state": meta["state"],
        "status": meta.get("status", "processing"),
        "result": meta.get("result"),
        "error": meta.get("error"),
    }


mcp = FastApiMCP(app, include_tags=["tool"])
mcp.mount_http(mount_path="/mcp")


@app.get("/")
async def root() -> RedirectResponse:
    """Root endpoint redirects to API documentation."""
    return RedirectResponse(url="/docs")
