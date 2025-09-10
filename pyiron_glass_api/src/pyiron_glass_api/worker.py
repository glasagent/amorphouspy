"""
Worker module for pyiron glass simulations.

This module contains the actual simulation logic that runs in separate processes,
isolated from the FastAPI server code to avoid unnecessary imports and potential
conflicts with signal handling.
"""

import os
import logging
from typing import Dict, Any

from .models import MeltquenchRequest


def setup_worker_logging(task_id: str) -> logging.Logger:
    """Setup logging for worker process."""
    logger = logging.getLogger(f"worker.{task_id}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"%(asctime)s - WORKER-{task_id} - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def meltquench_worker(task_id: str, request_dict: Dict[str, Any], task_store) -> None:
    """
    Synchronous worker function for meltquench simulation.
    This runs in a separate process to avoid blocking the event loop and handle pyiron's signal handling.

    Args:
        task_id (str): Unique identifier for the task
        request_dict (dict): Serialized meltquench parameters
        task_store: Shared multiprocessing dict for task state tracking
    """
    logger = setup_worker_logging(task_id)
    logger.info(f"Starting meltquench simulation for task {task_id}")
    
    # Reconstruct the request object from the dict
    request = MeltquenchRequest(**request_dict)
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
            fraction = value / 100.0 if sum(request.values) > 1.1 else value
            comp_parts.append(f"{fraction}{component}")

        composition = "-".join(comp_parts)
        logger.info(f"Task {task_id}: Generated composition string: {composition}")

        # Update task status
        task_dict = dict(task_store[task_id]) if task_id in task_store else {"state": "processing"}
        task_dict["status"] = "Creating structure"
        task_store[task_id] = task_dict
        logger.info(f"Task {task_id}: Creating structure")

        # Create pyiron project and generate structure
        pr = Project(f"meltquench_{task_id}")

        atoms_dict = get_structure_dict(
            comp=composition,
            min_distance=1.8,
            max_attempts_per_atom=10000,
            pyiron_project=pr,
        ).pull()
        logger.info(f"Task {task_id}: Structure dictionary created with {len(atoms_dict['atoms'])} atoms")

        structure = get_ase_structure(atoms_dict=atoms_dict, pyiron_project=pr)
        logger.info(f"Task {task_id}: ASE structure created")

        potential = generate_potential(atoms_dict=atoms_dict, pyiron_project=pr)
        logger.info(f"Task {task_id}: Potential generated")

        # Update task status
        task_dict = dict(task_store[task_id])
        task_dict["status"] = "Running meltquench simulation"
        task_store[task_id] = task_dict
        logger.info(f"Task {task_id}: Starting meltquench simulation")

        # Prepare a dedicated temporary working directory base (must exist beforehand)
        tmp_dir_base = os.path.abspath(f"lmp_tmp_directory_{task_id}")
        os.makedirs(tmp_dir_base, exist_ok=True)

        # Use simulation parameters from the request
        logger.info(f"Task {task_id}: Using heating_rate={request.heating_rate}, cooling_rate={request.cooling_rate}, n_print={request.n_print}")

        # Run meltquench simulation
        delayed = melt_quench_simulation(
            structure=structure,
            potential=potential,
            n_print=request.n_print,
            tmp_working_directory=tmp_dir_base,
            heating_rate=request.heating_rate,
            cooling_rate=request.cooling_rate,
            langevin=False,
            server_kwargs={},
        )

        # Execute the simulation
        logger.info(f"Task {task_id}: Executing simulation workflow")
        result = delayed.pull()
        logger.info(f"Task {task_id}: Simulation completed successfully")

        # Extract generic results from simulation output
        if not isinstance(result, dict):
            raise KeyError("Workflow output is not a dictionary")
        
        generic = result.get("result") or result.get("generic")
        if generic is None:
            raise KeyError("Missing simulation results ('result'/'generic') in workflow output")

        # Calculate final density
        V = np.mean(generic["volume"]) * 1e-24  # volume in cm³
        massTot = result["structure"].get_masses().sum() / units._Nav
        final_density = massTot / V
        logger.info(
            f"Task {task_id}: Final density calculated: {final_density:.3f} g/cm³"
        )

        # Store results
        task_store[task_id] = {
            "state": "complete",
            "status": "Completed",
            "result": {
                "composition": composition,
                "final_structure": str(result["structure"]),
                "mean_temperature": float(np.mean(generic["temperature"])),
                "final_density": float(final_density),
                "simulation_steps": len(generic["steps"]),
            }
        }
        
        logger.info(f"Task {task_id}: Results stored, simulation complete")

    except Exception as exc:
        logger.error(
            f"Task {task_id}: Simulation failed with error: {str(exc)}", exc_info=True
        )
        task_store[task_id] = {
            "state": "error",
            "status": "Failed",
            "error": str(exc)
        }
