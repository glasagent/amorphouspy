"""Worker module for amorphouspy simulations.

This module contains the actual simulation logic that runs in separate processes,
isolated from the FastAPI server code to avoid unnecessary imports and potential
conflicts with signal handling.
"""

import logging
from typing import Any

from .models import MeltquenchRequest


def setup_worker_logging(task_id: str) -> logging.Logger:
    """Set up logging for worker process.

    Args:
        task_id: The unique identifier for the task.

    Returns:
        Configured logger instance describing the worker process.
    """
    logger = logging.getLogger(f"worker.{task_id}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f"%(asctime)s - WORKER-{task_id} - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def meltquench_worker(task_id: str, request_dict: dict[str, Any], db_path: str, shared_project_dir: str) -> None:
    """Run synchronous meltquench simulation.

    This runs in a separate process to avoid blocking the event loop.

    Args:
        task_id: Unique identifier for the task.
        request_dict: Serialized meltquench parameters.
        db_path: Path to SQLite database for task store.
        shared_project_dir: Path to the shared project directory.
    """
    from pathlib import Path

    from .database import TaskStore

    logger = setup_worker_logging(task_id)
    logger.info(f"Starting meltquench simulation for task {task_id}")

    # Create task store instance for this worker process
    task_store = TaskStore(Path(db_path))

    # Reconstruct the request object from the dict
    request = MeltquenchRequest(**request_dict)
    logger.info(f"Request parameters: {request.model_dump()}")

    try:
        # Import amorphouspy modules (import here to avoid startup dependencies)
        import numpy as np
        from amorphouspy import (
            generate_potential,
            get_ase_structure,
            get_structure_dict,
            melt_quench_simulation,
        )
        from amorphouspy.workflows.structural_analysis import analyze_structure
        from executorlib import SingleNodeExecutor, get_item_from_future

        # Create composition string from request
        comp_parts = []
        for component, value in zip(request.components, request.values, strict=False):
            # Convert to fractions if percentages were provided
            fraction = value / 100.0 if sum(request.values) > 1.1 else value
            comp_parts.append(f"{fraction}{component}")

        composition = "-".join(comp_parts)
        logger.info(f"Task {task_id}: Generated composition string: {composition}")

        # Update task status
        current_task = task_store.get(task_id) or {"state": "processing"}
        current_task["status"] = "Creating structure"
        task_store.set(task_id, current_task)
        logger.info(f"Task {task_id}: Creating structure")

        # Use the shared project directory passed from the main process
        project_path = Path(shared_project_dir)
        logger.info(f"Task {task_id}: Using shared project directory: {project_path}")

        # Create executor for caching workflow results
        with SingleNodeExecutor(cache_directory=project_path) as exe:
            atoms_dict_future = exe.submit(
                get_structure_dict,
                composition=composition,
                # n_molecules=5000,  # Default number of molecules
                target_atoms=request.n_atoms,
            )
            # logger.info(f"Task {task_id}: Structure dictionary created with {len(atoms_dict['atoms'])} atoms")

            structure_future = exe.submit(
                get_ase_structure,
                atoms_dict=atoms_dict_future,
            )
            logger.info(f"Task {task_id}: ASE structure created")

            potential_future = exe.submit(
                generate_potential,
                atoms_dict=atoms_dict_future,
                potential_type=request.potential_type,
            )
            logger.info(f"Task {task_id}: Potential generated")

            # Update task status
            current_task = task_store.get(task_id) or {"state": "processing"}
            current_task["status"] = "Running meltquench simulation"
            task_store.set(task_id, current_task)
            logger.info(f"Task {task_id}: Starting meltquench simulation")

            # Use simulation parameters from the request
            logger.info(
                f"Task {task_id}: Using heating_rate={request.heating_rate}, cooling_rate={request.cooling_rate}, n_print={request.n_print}"
            )

            # Run meltquench simulation
            logger.info(f"Task {task_id}: Executing simulation workflow")
            result_future = exe.submit(
                melt_quench_simulation,
                structure=structure_future,
                potential=potential_future,
                n_print=request.n_print,
                # tmp_working_directory=str(tmp_dir_base), # note: if provided needs to be static - or prevents caching at executor level
                heating_rate=request.heating_rate,
                cooling_rate=request.cooling_rate,
                langevin=False,
                server_kwargs={},
            )
            logger.info(f"Task {task_id}: Simulation completed successfully")

            # Update task status for structural analysis
            current_task = task_store.get(task_id) or {"state": "processing"}
            current_task["status"] = "Running structural analysis"
            task_store.set(task_id, current_task)
            logger.info(f"Task {task_id}: Starting structural analysis")

            # Perform structural analysis on the final structure (includes density calculation)
            # final_structure = get_item_from_future(result_future, key="structure")
            # logger.info(f"Task {task_id}: Analyzing structure with {len(final_structure)} atoms")

            # Run structural analysis
            structural_data_future = exe.submit(
                analyze_structure, atoms=get_item_from_future(result_future, key="structure")
            )
            logger.info(f"Task {task_id}: Structural analysis completed successfully")
        
        result = result_future.result()
        structural_data = structural_data_future.result()

        # Debug: Check what fields are present in the structural_data object
        logger.info(f"Task {task_id}: StructureData type: {type(structural_data)}")
        if hasattr(structural_data, "model_fields"):
            logger.info(f"Task {task_id}: StructureData model fields: {list(structural_data.model_fields.keys())}")
        if hasattr(structural_data, "__dict__"):
            logger.info(f"Task {task_id}: StructureData attributes: {list(structural_data.__dict__.keys())}")

        # Use the structural data directly (it's now a Pydantic model with proper serialization)
        structural_summary = structural_data.model_dump() if hasattr(structural_data, "model_dump") else structural_data
        logger.info(f"Task {task_id}: Structural analysis data prepared")
        logger.info(
            f"Task {task_id}: Structural summary keys: {list(structural_summary.keys()) if isinstance(structural_summary, dict) else 'Not a dict'}"
        )

        # Store results including structural analysis
        current_task = task_store.get(task_id) or {}
        current_task.update(
            {
                "state": "complete",
                "status": "Completed",
                "result": {
                    "composition": composition,
                    "final_structure": result["structure"],  # Store ASE Atoms object directly
                    "mean_temperature": float(np.mean(result["result"]["temperature"])),
                    "simulation_steps": len(result["result"]["steps"]),
                    "structural_analysis": structural_summary,
                },
            }
        )
        task_store.set(task_id, current_task)

        logger.info(f"Task {task_id}: Results stored, simulation complete")

    except Exception as exc:
        logger.error(f"Task {task_id}: Simulation failed with error: {exc!s}", exc_info=True)
        current_task = task_store.get(task_id) or {}
        current_task.update({"state": "error", "status": "Failed", "error": str(exc)})
        task_store.set(task_id, current_task)
