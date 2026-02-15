"""Meltquench workflow for glass simulation.

This module contains the meltquench workflow function that uses executorlib
to submit different parts of the workflow with appropriate resources.

The workflow is structured as:
1. Structure generation and potential setup (lightweight, no special resources)
2. LAMMPS melt-quench simulation (compute-intensive, uses LAMMPS_CORES)
3. Structural analysis (post-processing, no special resources)
"""

import logging
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

import numpy as np
from amorphouspy import (
    generate_potential,
    get_ase_structure,
    get_structure_dict,
    melt_quench_simulation,
)
from amorphouspy.workflows.structural_analysis import analyze_structure

if TYPE_CHECKING:
    from executorlib.executor.base import BaseExecutor

logger = logging.getLogger(__name__)


def run_meltquench_workflow(
    executor: "BaseExecutor",
    components: list[str],
    values: list[float],
    n_atoms: int,
    potential_type: str,
    heating_rate: float,
    cooling_rate: float,
    n_print: int,
    lammps_resource_dict: dict[str, Any] | None = None,
    cache_key: str | None = None,
) -> Future[dict[str, Any]]:
    """Submit the complete meltquench workflow to the executor.

    This function submits multiple jobs to the executor with proper dependency
    tracking. Different parts of the workflow can use different resources:
    - Structure/potential generation: lightweight, default resources
    - LAMMPS simulation: compute-intensive, uses lammps_resource_dict

    Args:
        executor: The executorlib executor to submit jobs to.
        components: List of oxide components (e.g., ["SiO2", "Na2O", "B2O3"]).
        values: List of corresponding values (fractions or percentages).
        n_atoms: Target number of atoms in the simulation.
        potential_type: Type of interatomic potential to use.
        heating_rate: Heating rate in K/ps.
        cooling_rate: Cooling rate in K/ps.
        n_print: Number of steps between output prints.
        lammps_resource_dict: Resource dict for LAMMPS (e.g., {"cores": 4}).
        cache_key: Optional explicit cache key for the final workflow step.
            When set, the result can later be retrieved via
            ``get_future_from_cache(cache_directory, cache_key)``.

    Returns:
        Future that will resolve to the final result dictionary.
    """
    if lammps_resource_dict is None:
        lammps_resource_dict = {}

    # Build composition string from components and values
    comp_parts = []
    for component, value in zip(components, values, strict=False):
        fraction = value / 100.0 if sum(values) > 1.1 else value
        comp_parts.append(f"{fraction}{component}")
    composition = "-".join(comp_parts)
    logger.info("Submitting meltquench workflow for composition: %s", composition)

    # Step 1-3: Submit structure and potential generation (lightweight)
    atoms_dict_future = executor.submit(get_structure_dict, composition=composition, target_atoms=n_atoms)
    structure_future = executor.submit(get_ase_structure, atoms_dict=atoms_dict_future)
    potential_future = executor.submit(generate_potential, atoms_dict=atoms_dict_future, potential_type=potential_type)

    # Step 4: Submit LAMMPS melt-quench simulation (compute-intensive)
    meltquench_future = executor.submit(
        melt_quench_simulation,
        structure=structure_future,
        potential=potential_future,
        n_print=n_print,
        heating_rate=heating_rate,
        cooling_rate=cooling_rate,
        langevin=False,
        server_kwargs=lammps_resource_dict,
    )

    # Step 5: Submit structural analysis and result assembly
    final_resource_dict = {}
    if cache_key is not None:
        final_resource_dict["cache_key"] = cache_key
    return executor.submit(
        _assemble_results,
        composition=composition,
        meltquench_result=meltquench_future,
        resource_dict=final_resource_dict if final_resource_dict else {},
    )


def _assemble_results(composition: str, meltquench_result: dict[str, Any]) -> dict[str, Any]:
    """Perform structural analysis and assemble final results.

    Args:
        composition: Composition string.
        meltquench_result: Result from melt_quench_simulation.

    Returns:
        Final result dictionary with structural analysis.
    """
    final_structure = meltquench_result["structure"]
    structural_data = analyze_structure(atoms=final_structure)

    structural_summary = structural_data.model_dump() if hasattr(structural_data, "model_dump") else structural_data

    return {
        "composition": composition,
        "final_structure": final_structure,
        "mean_temperature": float(np.mean(meltquench_result["result"]["temperature"])),
        "simulation_steps": len(meltquench_result["result"]["steps"]),
        "structural_analysis": structural_summary,
    }
