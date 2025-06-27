"""
Asynchronous Task Management API using FastMCP

This module provides a FastMCP server for managing long-running asynchronous tasks.
It includes functionality to start tasks, check their status, and retrieve results.
Tasks are stored in-memory and can be in one of three states: processing, complete, or error.

Example usage:
    1. Start a task: start_sleep(5) -> returns task_id
    2. Check status: check(task_id) -> returns current status or result
"""

from uuid import uuid4
from typing import Dict
import json
import anyio
from fastmcp import FastMCP
from fastmcp.utilities import types

from models import MeltquenchRequest

mcp = FastMCP("long_runner")

# in-memory task store ---------------------------------
_task_store: Dict[str, dict] = {}


async def _worker(task_id: str, n: int) -> None:
    try:
        await anyio.sleep(n)  # do the expensive thing
        _task_store[task_id]["state"] = "complete"
        _task_store[task_id]["result"] = f"slept {n}s"
    except Exception as exc:
        _task_store[task_id]["state"] = "error"
        _task_store[task_id]["error"] = str(exc)


# tools ------------------------------------------------
@mcp.tool(description="Start task")
async def start_sleep(seconds: int) -> types.TextContent:
    task_id = str(uuid4())
    _task_store[task_id] = {"state": "processing"}
    anyio.create_task_group().start_soon(_worker, task_id, seconds)
    return types.TextContent(text=task_id)


@mcp.tool(description="Check status")
async def check(task_id: str) -> types.TextContent:
    meta = _task_store.get(task_id)
    if not meta:
        return types.TextContent(text="unknown task")
    if meta["state"] == "complete":
        return types.TextContent(text=meta["result"])
    if meta["state"] == "error":
        return types.TextContent(text=f"ERROR: {meta['error']}")
    return types.TextContent(text="processing")


async def _meltquench_worker(task_id: str, request: MeltquenchRequest) -> None:
    """
    Background worker function for meltquench simulation.

    Args:
        task_id (str): Unique identifier for the task
        request (MeltquenchRequest): Validated meltquench parameters
    """
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

        # Update task status
        _task_store[task_id]["status"] = "Creating structure"

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

        structure = get_ase_structure(
            atoms_dict=atoms_dict,
            pyiron_project=pr,
        )

        potential = generate_potential(
            atoms_dict=atoms_dict,
            pyiron_project=pr,
        )

        # Update task status
        _task_store[task_id]["status"] = "Running meltquench simulation"

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
        result = delayed.pull()

        # Calculate final density
        V = np.mean(result["generic"]["volume"]) * 1e-24  # volume in cm³
        massTot = result["structure"].get_masses().sum() / units._Nav
        final_density = massTot / V

        # Store results
        _task_store[task_id]["state"] = "complete"
        _task_store[task_id]["result"] = {
            "composition": composition,
            "final_structure": str(result["structure"]),
            "mean_temperature": float(np.mean(result["temperature"])),
            "final_density": float(final_density),
            "simulation_steps": len(result["steps"]),
        }

    except Exception as exc:
        _task_store[task_id]["state"] = "error"
        _task_store[task_id]["error"] = str(exc)


@mcp.tool(description="Start a meltquench simulation with specified composition")
async def start_meltquench(request_json: str) -> types.TextContent:
    """
    Start a new meltquench simulation task.

    Args:
        request_json (str): JSON string containing meltquench parameters

    Returns:
        types.TextContent: Task ID that can be used to check status

    Example:
        request = {
            "components": ["CaO", "Al2O3", "SiO2"],
            "values": [25.0, 25.0, 50.0],
            "unit": "mol",
            "n_molecules": 200,
            "density": 2.69
        }
        start_meltquench(json.dumps(request)) -> "task-id-string"
    """
    try:
        # Parse and validate the request
        request_data = json.loads(request_json)
        request = MeltquenchRequest(**request_data)

        task_id = str(uuid4())
        _task_store[task_id] = {"state": "processing", "status": "Initializing"}

        # Start the background task
        anyio.create_task_group().start_soon(_meltquench_worker, task_id, request)

        return types.TextContent(text=task_id)

    except json.JSONDecodeError as e:
        return types.TextContent(text=f"ERROR: Invalid JSON format - {str(e)}")
    except Exception as e:
        return types.TextContent(text=f"ERROR: {str(e)}")


@mcp.tool(description="Check the status of a running task")
async def check(task_id: str) -> types.TextContent:
    """
    Check the current status of a task by its ID.

    Args:
        task_id (str): Unique task identifier returned by start_sleep or start_meltquench

    Returns:
        types.TextContent: Current task status, result, or error message

    Possible return values:
        - "unknown task": Task ID not found
        - "processing": Task is still running
        - Task result: Task completed successfully (JSON for meltquench)
        - "ERROR: <message>": Task failed with error
    """
    meta = _task_store.get(task_id)
    if not meta:
        return types.TextContent(text="unknown task")
    if meta["state"] == "complete":
        # Return JSON for meltquench results, simple text for sleep results
        if isinstance(meta["result"], dict):
            return types.TextContent(text=json.dumps(meta["result"], indent=2))
        else:
            return types.TextContent(text=meta["result"])
    if meta["state"] == "error":
        return types.TextContent(text=f"ERROR: {meta['error']}")

    # Show detailed status for processing tasks
    status = meta.get("status", "processing")
    return types.TextContent(text=f"processing - {status}")
