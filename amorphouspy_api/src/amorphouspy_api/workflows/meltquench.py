"""Meltquench workflow for glass simulation.

This module contains the meltquench workflow function that can be
submitted to executorlib for local or SLURM execution.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def run_meltquench_workflow(
    components: list[str],
    values: list[float],
    n_atoms: int,
    potential_type: str,
    heating_rate: float,
    cooling_rate: float,
    n_print: int,
) -> dict[str, Any]:
    """Run the complete meltquench workflow.

    This function encapsulates the entire meltquench simulation workflow
    and is designed to be submitted via executorlib.

    Args:
        components: List of oxide components (e.g., ["SiO2", "Na2O", "B2O3"]).
        values: List of corresponding values (fractions or percentages).
        n_atoms: Target number of atoms in the simulation.
        potential_type: Type of interatomic potential to use.
        heating_rate: Heating rate in K/ps.
        cooling_rate: Cooling rate in K/ps.
        n_print: Number of steps between output prints.

    Returns:
        Dictionary containing simulation results and structural analysis.

    Raises:
        RuntimeError: If the simulation fails.
    """
    try:
        import numpy as np
        from amorphouspy import (
            generate_potential,
            get_ase_structure,
            get_structure_dict,
            melt_quench_simulation,
        )
        from amorphouspy.workflows.structural_analysis import analyze_structure

        # Build composition string from components and values
        comp_parts = []
        for component, value in zip(components, values, strict=False):
            # Convert to fractions if percentages were provided
            fraction = value / 100.0 if sum(values) > 1.1 else value
            comp_parts.append(f"{fraction}{component}")
        composition = "-".join(comp_parts)
        logger.info("Running meltquench for composition: %s", composition)

        # Create structure dictionary
        atoms_dict = get_structure_dict(
            composition=composition,
            target_atoms=n_atoms,
        )

        # Create ASE structure and potential
        structure = get_ase_structure(atoms_dict=atoms_dict)
        potential = generate_potential(atoms_dict=atoms_dict, potential_type=potential_type)
        logger.info("Structure created with %d atoms", len(structure))

        # Run meltquench simulation
        logger.info("Starting melt-quench simulation...")
        result = melt_quench_simulation(
            structure=structure,
            potential=potential,
            n_print=n_print,
            heating_rate=heating_rate,
            cooling_rate=cooling_rate,
            langevin=False,
            server_kwargs={},
        )
        logger.info("Simulation completed")

        # Perform structural analysis
        final_structure = result["structure"]
        structural_data = analyze_structure(atoms=final_structure)

        # Prepare output
        structural_summary = structural_data.model_dump() if hasattr(structural_data, "model_dump") else structural_data

        return {
            "composition": composition,
            "final_structure": result["structure"],
            "mean_temperature": float(np.mean(result["result"]["temperature"])),
            "simulation_steps": len(result["result"]["steps"]),
            "structural_analysis": structural_summary,
        }

    except Exception as e:
        logger.exception("Meltquench workflow failed")
        msg = f"Meltquench simulation failed: {e}"
        raise RuntimeError(msg) from e
