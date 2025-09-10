"""
Pyiron Glass Simulation API

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

from uuid import uuid4
from typing import Optional
import logging
import asyncio
import concurrent.futures
import multiprocessing
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi_mcp import FastApiMCP

from .models import MeltquenchRequest, MeltquenchResult
from .worker import meltquench_worker


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("glass_api.log")],
)
logger = logging.getLogger(__name__)

# Create a multiprocessing manager for shared data between processes
manager = multiprocessing.Manager()
_task_store = manager.dict()


async def _meltquench_worker(task_id: str, request: MeltquenchRequest) -> None:
    """
    Async wrapper for meltquench simulation that runs the synchronous worker in a process executor.

    Args:
        task_id (str): Unique identifier for the task
        request (MeltquenchRequest): Validated meltquench parameters
    """
    loop = asyncio.get_event_loop()
    
    # Convert request to dict for serialization across processes
    request_dict = request.model_dump()
    
    # Run the synchronous worker in a process executor to handle pyiron's signal handling
    with concurrent.futures.ProcessPoolExecutor() as executor:
        await loop.run_in_executor(executor, meltquench_worker, task_id, request_dict, _task_store)


# Create FastAPI app
app = FastAPI(
    title="Pyiron Glass Simulation API",
    description="API for managing long-running glass simulation tasks using pyiron-glass",
    version="0.1.0",
)


@app.post("/check_cached_result", tags=["tool"])
async def check_cached_result(request: MeltquenchRequest) -> Optional[MeltquenchResult]:
    """
    Check if a result for the given meltquench request is already available in cache.
    
    Args:
        request (MeltquenchRequest): The meltquench parameters to check
        
    Returns:
        MeltquenchResult or None: The cached result if available, otherwise None
        
    Note:
        This is a placeholder implementation. The actual caching logic should be 
        implemented at the pyiron level, not at the API level.
    """
    try:
        logger.info(f"Checking for cached result with components: {request.components}")
        
        # TODO: Implement actual cache checking at pyiron level
        # For now, always return None (no cached results)
        logger.info("No cached result found (placeholder implementation)")
        return None
            
    except Exception as e:
        logger.error(f"Error checking cached result: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# FastAPI endpoints
@app.post("/submit_meltquench", tags=["tool"])
async def submit_meltquench(request: MeltquenchRequest):
    """Start a new meltquench simulation task"""
    try:
        task_id = str(uuid4())
        logger.info(f"Creating new meltquench task with ID: {task_id}")

        _task_store[task_id] = {"state": "processing", "status": "Initializing"}

        # Always run as background task using process executor
        asyncio.create_task(_meltquench_worker(task_id, request))

        return {"task_id": task_id, "status": "started"}
    except Exception as e:
        logger.error(f"Error starting meltquench task: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/check/{task_id}", tags=["tool"])
async def check(task_id: str):
    """Check the current status of a simulation task by its ID"""
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
mcp.mount_sse(mount_path='/mcp')


@app.get("/")
async def root():
    """Root endpoint redirects to API documentation"""
    return RedirectResponse(url="/docs")
