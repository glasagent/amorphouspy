"""Worker module for pyiron glass simulations.

This module contains the actual simulation logic that runs in separate processes,
isolated from the FastAPI server code to avoid unnecessary imports and potential
conflicts with signal handling.
"""

import logging
from typing import TYPE_CHECKING, Any

from .models import MeltquenchRequest, ViscosityRequest, ViscosityResult

if TYPE_CHECKING:
    from ase import Atoms


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

    This runs in a separate process to avoid blocking the event loop and handle pyiron's signal handling.

    Args:
        task_id: Unique identifier for the task.
        request_dict: Serialized meltquench parameters.
        db_path: Path to SQLite database for task store.
        shared_project_dir: Path to the shared project directory.
    """
    from pathlib import Path

    from pyiron_base import state

    from .database import TaskStore

    logger = setup_worker_logging(task_id)
    logger.info("[WORKER PROCESS] pyiron state.settings.configuration: %s", state.settings.configuration)
    logger.info(f"Starting meltquench simulation for task {task_id}")

    # Create task store instance for this worker process
    task_store = TaskStore(Path(db_path))

    # Reconstruct the request object from the dict
    request = MeltquenchRequest(**request_dict)
    logger.info(f"Request parameters: {request.model_dump()}")

    try:
        # Import pyiron_glass modules (import here to avoid startup dependencies)
        import numpy as np
        from pyiron_base import Project
        from pyiron_glass import (
            generate_potential,
            get_ase_structure,
            get_structure_dict,
            melt_quench_simulation,
        )
        from pyiron_glass.workflows.structural_analysis import analyze_structure

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

        # Create shared pyiron project
        pr = Project(str(project_path))

        atoms_dict = get_structure_dict(
            composition=composition,
            # n_molecules=5000,  # Default number of molecules
            target_atoms=request.n_atoms,
            pyiron_project=pr,
        ).pull()
        logger.info(f"Task {task_id}: Structure dictionary created with {len(atoms_dict['atoms'])} atoms")

        structure = get_ase_structure(atoms_dict=atoms_dict, pyiron_project=pr)
        logger.info(f"Task {task_id}: ASE structure created")

        potential = generate_potential(atoms_dict=atoms_dict, potential_type=request.potential_type, pyiron_project=pr)
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
        result = melt_quench_simulation(
            structure=structure,
            potential=potential,
            n_print=request.n_print,
            pyiron_project=pr,
            # tmp_working_directory=str(tmp_dir_base), # note: if provided needs to be static - or prevents caching at pyiron level
            heating_rate=request.heating_rate,
            cooling_rate=request.cooling_rate,
            langevin=False,
            server_kwargs={},
        ).pull()
        logger.info(f"Task {task_id}: Simulation completed successfully")

        # Update task status for structural analysis
        current_task = task_store.get(task_id) or {"state": "processing"}
        current_task["status"] = "Running structural analysis"
        task_store.set(task_id, current_task)
        logger.info(f"Task {task_id}: Starting structural analysis")

        # Perform structural analysis on the final structure (includes density calculation)
        final_structure = result["structure"]
        logger.info(f"Task {task_id}: Analyzing structure with {len(final_structure)} atoms")

        # Run structural analysis (decorated with @job, needs pyiron_project and .pull())
        structural_data = analyze_structure(atoms=final_structure, pyiron_project=pr).pull()
        logger.info(f"Task {task_id}: Structural analysis completed successfully")

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


def viscosity_worker(task_id: str, request_dict: dict[str, Any], db_path: str, shared_project_dir: str) -> None:
    """Run synchronous viscosity simulation, optionally preceded by meltquench.

    This runs in a separate process to avoid blocking the event loop and handle pyiron's signal handling.

    Args:
        task_id: Unique identifier for the task.
        request_dict: Serialized viscosity parameters.
        db_path: Path to SQLite database for task store.
        shared_project_dir: Path to the shared project directory.
    """
    from pathlib import Path

    from pyiron_base import Project
    from pyiron_glass import (
        generate_potential,
        get_ase_structure,
        get_structure_dict,
        melt_quench_simulation,
    )
    from pyiron_glass.workflows.viscosity import get_viscosity, viscosity_simulation

    from .database import TaskStore
    from .models import validate_atoms

    logger = setup_worker_logging(task_id)
    logger.info(f"Starting viscosity simulation for task {task_id}")

    # Create task store instance for this worker process
    task_store = TaskStore(Path(db_path))

    # Reconstruct the request object from the dict
    request = ViscosityRequest(**request_dict)
    logger.info(f"Viscosity request parameters: {request.model_dump()}")

    try:
        # Create shared pyiron project
        project_path = Path(shared_project_dir)
        logger.info(f"Task {task_id}: Using shared project directory: {project_path}")
        pr = Project(str(project_path))

        composition: str | None = None

        # Normalize and sort temperatures: always start from highest and work down
        temperatures = sorted(request.temperatures or [], reverse=True)
        logger.info("Task %s: Viscosity temperatures (high->low): %s", task_id, temperatures)

        viscosities: list[float] = []
        all_max_lags: list[list[float]] = []
        sim_steps: list[int] = []
        lag_times_ps: list[list[float]] = []
        sacf_data: list[dict[str, list[float]]] = []
        viscosity_running: list[list[float]] = []

        # Determine starting structure: either from meltquench or from user-provided structure
        if request.meltquench_request is not None:
            logger.info("Task %s: Generating initial structure via meltquench workflow", task_id)

            # Create composition string from nested meltquench request
            mq_req = request.meltquench_request
            comp_parts: list[str] = []
            total_values = sum(mq_req.values)
            for component, value in zip(mq_req.components, mq_req.values, strict=False):
                fraction = value / 100.0 if total_values > 1.1 else value
                comp_parts.append(f"{fraction}{component}")
            composition = "-".join(comp_parts)
            logger.info("Task %s: Generated composition string for viscosity: %s", task_id, composition)

            # Update task status
            current_task = task_store.get(task_id) or {"state": "processing"}
            current_task["status"] = "Creating structure for viscosity"
            task_store.set(task_id, current_task)

            # Generate starting structure via pyiron_glass helpers
            atoms_dict = get_structure_dict(
                composition=composition,
                target_atoms=mq_req.n_atoms,
                pyiron_project=pr,
            ).pull()
            logger.info(
                "Task %s: Structure dictionary created for viscosity with %d atoms",
                task_id,
                len(atoms_dict["atoms"]),
            )

            structure_current = get_ase_structure(atoms_dict=atoms_dict, pyiron_project=pr)
            logger.info("Task %s: Initial ASE structure for viscosity created", task_id)

            potential = generate_potential(
                atoms_dict=atoms_dict, potential_type=mq_req.potential_type, pyiron_project=pr
            )
            logger.info("Task %s: Potential for viscosity generated", task_id)

            # Sequential cooling: from 5000 K to highest T, then stepwise downwards
            for idx, T in enumerate(temperatures):
                temp_high = 5000.0 if idx == 0 else temperatures[idx - 1]

                # Pre-cool / equilibrate at this temperature via melt-quench style protocol
                current_task = task_store.get(task_id) or {"state": "processing"}
                current_task["status"] = f"Cooling structure to {T} K for viscosity"
                task_store.set(task_id, current_task)
                logger.info("Task %s: Cooling from %.1f K to %.1f K before viscosity run", task_id, temp_high, T)

                mq_result = melt_quench_simulation(
                    structure=structure_current,
                    potential=potential,
                    temperature_high=float(temp_high),
                    temperature_low=float(T),
                    timestep=1.0,
                    heating_rate=float(mq_req.heating_rate),
                    cooling_rate=float(mq_req.cooling_rate),
                    n_print=mq_req.n_print,
                    langevin=False,
                    server_kwargs={},
                    pyiron_project=pr,
                ).pull()

                # Save the cooled structure at this temperature
                structure_at_T: Atoms = mq_result["structure"]
                logger.info(
                    "Task %s: Cooling to %.1f K completed, structure has %d atoms",
                    task_id,
                    T,
                    len(structure_at_T),
                )

                # Run viscosity simulation at this temperature
                current_task = task_store.get(task_id) or {"state": "processing"}
                current_task["status"] = f"Running viscosity simulation at {T} K"
                task_store.set(task_id, current_task)
                logger.info("Task %s: Starting viscosity simulation at %.1f K", task_id, T)

                visc_job = viscosity_simulation(
                    structure=structure_at_T,
                    potential=potential,
                    temperature_sim=float(T),
                    timestep=float(request.timestep),
                    production_steps=int(request.n_timesteps),
                    n_print=int(request.n_print),
                    langevin=False,
                    seed=12345,
                    server_kwargs={},
                    pyiron_project=pr,
                )
                visc_result = visc_job.pull()
                logger.info("Task %s: Viscosity simulation at %.1f K completed", task_id, T)

                # Post-process viscosity using Green-Kubo analysis
                viscosity_data = get_viscosity(
                    visc_result,
                    timestep=float(request.timestep),
                    max_lag=request.max_lag,
                )
                logger.info(
                    "Task %s: Viscosity post-processing at %.1f K completed, eta=%.3e Pa·s",
                    task_id,
                    viscosity_data["temperature"],
                    viscosity_data["viscosity"],
                )

                # Debug: Log available fields from get_viscosity
                logger.info("Task %s: get_viscosity returned fields: %s", task_id, list(viscosity_data.keys()))

                viscosities.append(float(viscosity_data["viscosity"]))
                all_max_lags.append(float(viscosity_data["max_lag"]))
                sim_steps.append(int(request.n_timesteps))

                # Store averaged SACF data for visualization
                lag_time = viscosity_data.get("lag_time_ps", [])
                sacf_avg = viscosity_data.get("sacf", [])
                visc_running = viscosity_data.get("viscosity_running", [])

                logger.info(
                    "Task %s: Collected visualization data at %.1f K - lag_time: %d pts, sacf: %d pts, visc_running: %d pts",
                    task_id,
                    T,
                    len(lag_time),
                    len(sacf_avg),
                    len(visc_running),
                )

                lag_times_ps.append(lag_time)
                sacf_data.append(sacf_avg)  # Now storing single averaged SACF
                viscosity_running.append(visc_running)

                # Use the pre-viscosity cooled structure as the starting point for the next cooling step
                structure_current = structure_at_T

        else:
            logger.info("Task %s: Using user-provided initial structure for viscosity", task_id)
            # Validate and reconstruct ASE Atoms from the provided structure
            atoms = validate_atoms(request.initial_structure)
            if atoms is None:
                msg = "initial_structure must not be None when meltquench_request is not provided"
                raise ValueError(msg)
            structure_current = atoms

            # Generate potential using the user-provided structure
            atoms_dict = {
                "atoms": [
                    {"element": str(sym), "position": pos.tolist()}
                    for sym, pos in zip(
                        structure_current.get_chemical_symbols(), structure_current.get_positions(), strict=False
                    )
                ]
            }
            potential = generate_potential(
                atoms_dict=atoms_dict, potential_type=request.potential_type, pyiron_project=pr
            )
            logger.info("Task %s: Potential for viscosity (custom structure) generated", task_id)

            # For custom structures, run viscosity simulations independently at each temperature
            for T in temperatures:
                current_task = task_store.get(task_id) or {"state": "processing"}
                current_task["status"] = f"Running viscosity simulation at {T} K"
                task_store.set(task_id, current_task)
                logger.info("Task %s: Starting viscosity simulation at %.1f K (custom structure)", task_id, T)

                visc_job = viscosity_simulation(
                    structure=structure_current,
                    potential=potential,
                    temperature_sim=float(T),
                    timestep=float(request.timestep),
                    production_steps=int(request.n_timesteps),
                    n_print=int(request.n_print),
                    langevin=False,
                    seed=12345,
                    server_kwargs={},
                    pyiron_project=pr,
                )
                visc_result = visc_job.pull()
                logger.info("Task %s: Viscosity simulation at %.1f K completed", task_id, T)

                viscosity_data = get_viscosity(
                    visc_result,
                    timestep=float(request.timestep),
                    max_lag=request.max_lag,
                )
                logger.info(
                    "Task %s: Viscosity post-processing at %.1f K completed, eta=%.3e Pa·s",
                    task_id,
                    viscosity_data["temperature"],
                    viscosity_data["viscosity"],
                )

                # Debug: Log available fields from get_viscosity
                logger.info("Task %s: get_viscosity returned fields: %s", task_id, list(viscosity_data.keys()))

                viscosities.append(float(viscosity_data["viscosity"]))
                all_max_lags.append(float(viscosity_data["max_lag"]))
                sim_steps.append(int(request.n_timesteps))

                # Store averaged SACF data for visualization
                lag_time = viscosity_data.get("lag_time_ps", [])
                sacf_avg = viscosity_data.get("sacf", [])
                visc_running = viscosity_data.get("viscosity_running", [])

                logger.info(
                    "Task %s: Collected visualization data at %.1f K - lag_time: %d pts, sacf: %d pts, visc_running: %d pts",
                    task_id,
                    T,
                    len(lag_time),
                    len(sacf_avg),
                    len(visc_running),
                )

                lag_times_ps.append(lag_time)
                sacf_data.append(sacf_avg)  # Now storing single averaged SACF
                viscosity_running.append(visc_running)

        # Prepare multi-temperature result model
        result_model = ViscosityResult(
            composition=composition,
            temperatures=temperatures,
            viscosities=viscosities,
            max_lag=all_max_lags,
            simulation_steps=sim_steps,
            lag_times_ps=lag_times_ps,
            sacf_data=sacf_data,
            viscosity_running=viscosity_running,
        )

        # Store results
        current_task = task_store.get(task_id) or {}
        current_task.update(
            {
                "state": "complete",
                "status": "Completed",
                "result": result_model.model_dump(),
            }
        )
        task_store.set(task_id, current_task)

        logger.info("Task %s: Viscosity results stored, simulation complete", task_id)

    except Exception as exc:
        logger.error("Task %s: Viscosity simulation failed with error: %s", task_id, exc, exc_info=True)
        current_task = task_store.get(task_id) or {}
        current_task.update({"state": "error", "status": "Failed", "error": str(exc)})
        task_store.set(task_id, current_task)
