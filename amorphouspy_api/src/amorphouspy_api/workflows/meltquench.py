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

from amorphouspy.pipelines.meltquench import generate_structure as _generate_structure
from amorphouspy.pipelines.meltquench import run_melt_quench as _run_melt_quench

if TYPE_CHECKING:
    from pydantic import BaseModel

    from amorphouspy_api.models import JobSubmission

logger = logging.getLogger(__name__)


def generate_structure(submission: "JobSubmission", config: "BaseModel", result: dict) -> dict:
    """Generate initial structure and potential from composition.

    Returns a dict with ``atoms_dict``, ``structure`` (ASE Atoms as dict),
    and ``potential``.
    """
    return _generate_structure(
        composition=submission.composition.root,
        n_atoms=submission.simulation.n_atoms,
        potential_type=submission.potential,
        density=submission.simulation.target_density,
        structure_seed=submission.simulation.structure_seed,
        electrostatics_config=submission.electrostatics.to_electrostatics_config(),
    )


def run_melt_quench(submission: "JobSubmission", config: "BaseModel", result: dict) -> dict:
    """Run the LAMMPS melt-quench simulation.

    Expects ``result`` to contain the output of ``generate_structure``
    (keys: ``structure``, ``potential``).

    Returns a dict with ``final_structure``, ``mean_temperature``,
    ``simulation_steps``, and ``composition``.
    """
    from amorphouspy_api.executor import get_lammps_server_kwargs

    mq_result = _run_melt_quench(
        structure=result["structure_generation"]["structure"],
        potential=result["structure_generation"]["potential"],
        potential_type=submission.potential,
        heating_rate=int(submission.simulation.quench_rate * 100),
        cooling_rate=int(submission.simulation.quench_rate),
        timestep=submission.simulation.timestep,
        temperature_high=submission.simulation.melt_temperature,
        temperature_low=300.0,
        equilibration_steps=submission.simulation.equilibration_steps,
        n_averaging_frames=getattr(submission.simulation, "n_averaging_frames", 100),
        server_kwargs=get_lammps_server_kwargs(),
    )
    mq_result["composition"] = submission.composition.root
    return mq_result
