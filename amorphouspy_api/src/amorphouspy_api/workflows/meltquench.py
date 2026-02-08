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

if TYPE_CHECKING:
    from ase import Atoms
    from executorlib.executor.single import TestClusterExecutor

logger = logging.getLogger(__name__)


def run_meltquench_workflow(
    executor: "TestClusterExecutor",
    components: list[str],
    values: list[float],
    n_atoms: int,
    potential_type: str,
    heating_rate: float,
    cooling_rate: float,
    n_print: int,
    lammps_resource_dict: dict[str, Any] | None = None,
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

    Returns:
        Future that will resolve to the final result dictionary.
    """
    if lammps_resource_dict is None:
        lammps_resource_dict = {}

    # Build composition string from components and values
    comp_parts = []
    for component, value in zip(components, values, strict=False):
        # Convert to fractions if percentages were provided
        fraction = value / 100.0 if sum(values) > 1.1 else value
        comp_parts.append(f"{fraction}{component}")
    composition = "-".join(comp_parts)
    logger.info("Submitting meltquench workflow for composition: %s", composition)

    # Step 1: Submit structure generation (lightweight)
    atoms_dict_future = executor.submit(
        _get_structure_dict_wrapper,
        composition=composition,
        target_atoms=n_atoms,
    )

    # Step 2: Submit ASE structure creation (depends on atoms_dict)
    structure_future = executor.submit(
        _get_ase_structure_wrapper,
        atoms_dict=atoms_dict_future,
    )

    # Step 3: Submit potential generation (depends on atoms_dict)
    potential_future = executor.submit(
        _generate_potential_wrapper,
        atoms_dict=atoms_dict_future,
        potential_type=potential_type,
    )

    # Step 4: Submit LAMMPS melt-quench simulation (compute-intensive)
    # This uses the lammps_resource_dict for LAMMPS-specific settings
    meltquench_future = executor.submit(
        _run_meltquench_simulation,
        structure=structure_future,
        potential=potential_future,
        n_print=n_print,
        heating_rate=heating_rate,
        cooling_rate=cooling_rate,
        server_kwargs=lammps_resource_dict,
    )

    # Step 5: Submit structural analysis and result assembly (lightweight)
    return executor.submit(
        _assemble_results,
        composition=composition,
        meltquench_result=meltquench_future,
    )


def _get_structure_dict_wrapper(
    composition: str,
    target_atoms: int,
) -> dict[str, Any]:
    """Create structure dictionary for the given composition.

    Args:
        composition: Composition string (e.g., "0.25CaO-0.30Al2O3-0.45SiO2").
        target_atoms: Target number of atoms.

    Returns:
        Structure dictionary.
    """
    from amorphouspy import get_structure_dict

    return get_structure_dict(
        composition=composition,
        target_atoms=target_atoms,
    )


def _get_ase_structure_wrapper(atoms_dict: dict[str, Any]) -> "Atoms":
    """Create ASE Atoms object from structure dictionary.

    Args:
        atoms_dict: Structure dictionary from get_structure_dict.

    Returns:
        ASE Atoms object.
    """
    from amorphouspy import get_ase_structure

    return get_ase_structure(atoms_dict=atoms_dict)


def _generate_potential_wrapper(
    atoms_dict: dict[str, Any],
    potential_type: str,
) -> dict[str, Any]:
    """Generate interatomic potential for the given structure.

    Args:
        atoms_dict: Structure dictionary from get_structure_dict.
        potential_type: Type of interatomic potential.

    Returns:
        Potential dictionary.
    """
    from amorphouspy import generate_potential

    return generate_potential(
        atoms_dict=atoms_dict,
        potential_type=potential_type,
    )


def _run_meltquench_simulation(
    structure: "Atoms",
    potential: dict[str, Any],
    n_print: int,
    heating_rate: float,
    cooling_rate: float,
    server_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Run the LAMMPS melt-quench simulation.

    Args:
        structure: ASE Atoms object.
        potential: Potential dictionary.
        n_print: Print interval.
        heating_rate: Heating rate in K/ps.
        cooling_rate: Cooling rate in K/ps.
        server_kwargs: LAMMPS server kwargs (e.g., cores).

    Returns:
        Simulation result dictionary.
    """
    import logging

    from amorphouspy import melt_quench_simulation

    logger = logging.getLogger(__name__)
    logger.info("Starting LAMMPS melt-quench simulation with %d atoms", len(structure))

    result = melt_quench_simulation(
        structure=structure,
        potential=potential,
        n_print=n_print,
        heating_rate=heating_rate,
        cooling_rate=cooling_rate,
        langevin=False,
        server_kwargs=server_kwargs,
    )

    logger.info("LAMMPS simulation completed")
    return result


def _assemble_results(
    composition: str,
    meltquench_result: dict[str, Any],
) -> dict[str, Any]:
    """Perform structural analysis and assemble final results.

    Args:
        composition: Composition string.
        meltquench_result: Result from melt_quench_simulation.

    Returns:
        Final result dictionary with structural analysis.
    """
    import logging

    import numpy as np
    from amorphouspy.workflows.structural_analysis import analyze_structure

    logger = logging.getLogger(__name__)
    logger.info("Performing structural analysis")

    final_structure = meltquench_result["structure"]
    structural_data = analyze_structure(atoms=final_structure)

    # Prepare output
    structural_summary = structural_data.model_dump() if hasattr(structural_data, "model_dump") else structural_data

    return {
        "composition": composition,
        "final_structure": final_structure,
        "mean_temperature": float(np.mean(meltquench_result["result"]["temperature"])),
        "simulation_steps": len(meltquench_result["result"]["steps"]),
        "structural_analysis": structural_summary,
    }
