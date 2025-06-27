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
from typing import Dict
import logging
import anyio
from fastapi import FastAPI, HTTPException
from fastapi_mcp import FastApiMCP

from models import MeltquenchRequest


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("glass_api.log")],
)
logger = logging.getLogger(__name__)

# In-memory task store for tracking simulation states and results
_task_store: Dict[str, dict] = {}


async def _meltquench_worker(task_id: str, request: MeltquenchRequest) -> None:
    """
    Background worker function for meltquench simulation.

    Args:
        task_id (str): Unique identifier for the task
        request (MeltquenchRequest): Validated meltquench parameters
    """
    logger.info(f"Starting meltquench simulation for task {task_id}")
    logger.info(f"Request parameters: {request.model_dump()}")

    try:
        # Import pyiron_glass modules (import here to avoid startup dependencies)
        import numpy as np
        from pyiron_base import Project
        from pyiron_glass import (
            melt_quench_simulation,
            generate_potential,
            get_ase_structure,
            get_structure_dict,
        )
        from ase import units

        # Create composition string from request
        comp_parts = []
        for component, value in zip(request.components, request.values):
            # Convert to fractions if percentages were provided
            if sum(request.values) > 1.1:  # Likely percentages
                fraction = value / 100.0
            else:
                fraction = value
            comp_parts.append(f"{fraction}{component}")

        composition = "-".join(comp_parts)
        logger.info(f"Task {task_id}: Generated composition string: {composition}")

        # Update task status
        _task_store[task_id]["status"] = "Creating structure"
        logger.info(f"Task {task_id}: Creating structure")

        # Create pyiron project and generate structure
        pr = Project(f"meltquench_{task_id}")

        atoms_dict = get_structure_dict(
            comp=composition,
            n_molecules=request.n_molecules,
            density=request.density,
            min_distance=1.8,
            max_attempts_per_atom=10000,
            pyiron_project=pr,
        )
        logger.info(
            f"Task {task_id}: Structure dictionary created with {len(atoms_dict['atoms'])} atoms"
        )

        structure = get_ase_structure(
            atoms_dict=atoms_dict,
            pyiron_project=pr,
        )
        logger.info(f"Task {task_id}: ASE structure created")

        potential = generate_potential(
            atoms_dict=atoms_dict,
            pyiron_project=pr,
        )
        logger.info(f"Task {task_id}: Potential generated")

        # Update task status
        _task_store[task_id]["status"] = "Running meltquench simulation"
        logger.info(f"Task {task_id}: Starting meltquench simulation")

        # Run meltquench simulation
        delayed = melt_quench_simulation(
            structure=structure,
            potential=potential,
            temperature_high=request.temperature_high,
            temperature_low=request.temperature_low,
            n_print=1000,
            working_directory=f"lmp_tmp_directory_{task_id}",
            heating_rate=int(1e14),
            cooling_rate=int(1e14),
            langevin=False,
            pyiron_project=pr,
        )

        # Execute the simulation
        logger.info(f"Task {task_id}: Executing simulation workflow")
        result = delayed.pull()
        logger.info(f"Task {task_id}: Simulation completed successfully")

        # Calculate final density
        V = np.mean(result["generic"]["volume"]) * 1e-24  # volume in cm³
        massTot = result["structure"].get_masses().sum() / units._Nav
        final_density = massTot / V
        logger.info(
            f"Task {task_id}: Final density calculated: {final_density:.3f} g/cm³"
        )

        # Store results
        _task_store[task_id]["state"] = "complete"
        _task_store[task_id]["result"] = {
            "composition": composition,
            "final_structure": str(result["structure"]),
            "mean_temperature": float(np.mean(result["temperature"])),
            "final_density": float(final_density),
            "simulation_steps": len(result["steps"]),
        }
        logger.info(f"Task {task_id}: Results stored, simulation complete")

    except Exception as exc:
        logger.error(
            f"Task {task_id}: Simulation failed with error: {str(exc)}", exc_info=True
        )
        _task_store[task_id]["state"] = "error"
        _task_store[task_id]["error"] = str(exc)


# Create FastAPI app
app = FastAPI(
    title="Pyiron Glass Simulation API",
    description="API for managing long-running glass simulation tasks using pyiron-glass",
    version="0.1.0",
)


# FastAPI endpoints
@app.post("/submit_meltquench")
async def submit_meltquench(request: MeltquenchRequest):
    """Start a new meltquench simulation task"""
    try:
        task_id = str(uuid4())
        logger.info(f"Creating new meltquench task with ID: {task_id}")

        _task_store[task_id] = {"state": "processing", "status": "Initializing"}

        # Start the background task
        anyio.create_task_group().start_soon(_meltquench_worker, task_id, request)

        return {"task_id": task_id, "status": "started"}

    except Exception as e:
        logger.error(f"Error starting meltquench task: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/check/{task_id}")
async def check(task_id: str):
    """Check the current status of a simulation task by its ID"""
    logger.debug(f"Checking status for task: {task_id}")
    meta = _task_store.get(task_id)
    if not meta:
        logger.warning(f"Unknown task ID requested: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task_id,
        "state": meta["state"],
        "status": meta.get("status", "processing"),
        "result": meta.get("result"),
        "error": meta.get("error"),
    }


mcp = FastApiMCP(app)
mcp.mount()


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Pyiron Glass Simulation API",
        "endpoints": {
            "submit_meltquench": "POST /submit_meltquench",
            "check_task": "GET /check/{task_id}",
        },
    }

if __name__ == "__main__":
    import uvicorn
    # For WSL2, you might need to use the WSL IP or 0.0.0.0
    uvicorn.run(app, host="0.0.0.0", port=8002)
