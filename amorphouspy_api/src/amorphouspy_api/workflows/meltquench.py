"""Meltquench workflow for glass simulation.

This module contains the individual pipeline steps that use executorlib
to submit work with appropriate resources.

The workflow is structured as:
1. ``generate_structure`` — structure generation and potential setup (lightweight)
2. ``run_melt_quench`` — LAMMPS melt-quench simulation (compute-intensive)

Analysis (structural, viscosity, etc.) happens *after* the melt-quench
completes, via additional step functions registered in ``workflows.analyses``.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
from amorphouspy import (
    generate_potential,
    get_ase_structure,
    get_structure_dict,
    melt_quench_simulation,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from amorphouspy_api.models import JobSubmission

logger = logging.getLogger(__name__)


def generate_structure(submission: "JobSubmission", config: "BaseModel", result: dict) -> dict:
    """Generate initial structure and potential from composition.

    Returns a dict with ``atoms_dict``, ``structure`` (ASE Atoms as dict),
    and ``potential``.
    """
    composition = submission.composition.root
    n_atoms = submission.simulation.n_atoms
    potential_type = submission.potential.value

    atoms_dict = get_structure_dict(composition=composition, target_atoms=n_atoms)
    structure = get_ase_structure(atoms_dict=atoms_dict)
    potential = generate_potential(atoms_dict=atoms_dict, potential_type=potential_type)

    return {
        "atoms_dict": atoms_dict,
        "structure": structure,
        "potential": potential,
    }


def run_melt_quench(submission: "JobSubmission", config: "BaseModel", result: dict) -> dict:
    """Run the LAMMPS melt-quench simulation.

    Expects ``result`` to contain the output of ``generate_structure``
    (keys: ``structure``, ``potential``).

    Returns a dict with ``final_structure``, ``mean_temperature``,
    ``simulation_steps``, and ``composition``.
    """
    from amorphouspy_api.executor import get_lammps_server_kwargs

    structure = result["structure_generation"]["structure"]
    potential = result["structure_generation"]["potential"]

    mq = melt_quench_simulation(
        structure=structure,
        potential=potential,
        n_print=1000,
        heating_rate=int(submission.simulation.quench_rate * 100),
        cooling_rate=int(submission.simulation.quench_rate),
        langevin=False,
        server_kwargs=get_lammps_server_kwargs(),
    )

    return {
        "composition": submission.composition.root,
        "final_structure": mq["structure"],
        "mean_temperature": float(np.mean(mq["result"]["temperature"])),
        "simulation_steps": len(mq["result"]["steps"]),
    }
