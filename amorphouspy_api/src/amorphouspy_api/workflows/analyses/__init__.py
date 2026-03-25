"""Post-melt-quench analysis functions.

Each function receives ``(submission, config, mq_result)`` and returns a
dict of results.  The ``ANALYSES`` mapping connects the API type strings
(``"structure"``, ``"viscosity"``, …) to these functions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

    from amorphouspy_api.models import JobSubmission, ViscosityAnalysis

AnalysisFn = Callable[["JobSubmission", "BaseModel", dict], dict]


def run_structure(submission: JobSubmission, config: BaseModel, mq_result: dict) -> dict:
    """Structural analysis (RDF, coordination, bond angles) on the quenched glass."""
    from amorphouspy.workflows.structural_analysis import analyze_structure

    data = analyze_structure(atoms=mq_result["final_structure"])
    return data.model_dump() if hasattr(data, "model_dump") else data


def run_viscosity(submission: JobSubmission, config: BaseModel, mq_result: dict) -> dict:
    """Multi-temperature viscosity analysis on the quenched glass."""
    from amorphouspy import generate_potential, get_structure_dict

    from amorphouspy_api.executor import get_lammps_resource_dict
    from amorphouspy_api.workflows.analyses.viscosity import run_viscosity_workflow

    cfg: ViscosityAnalysis = config  # type: ignore[assignment]

    atoms_dict = get_structure_dict(
        composition=submission.composition.root,
        target_atoms=submission.simulation.n_atoms,
    )
    potential = generate_potential(
        atoms_dict=atoms_dict,
        potential_type=submission.potential.value,
    )

    return run_viscosity_workflow(
        structure=mq_result["final_structure"],
        potential=potential,
        temperatures=cfg.temperatures,
        heating_rate=int(submission.simulation.quench_rate * 100),
        cooling_rate=int(submission.simulation.quench_rate),
        timestep=cfg.timestep,
        n_timesteps=cfg.n_timesteps,
        n_print=cfg.n_print,
        max_lag=cfg.max_lag,
        lammps_resource_dict=get_lammps_resource_dict(),
    )


ANALYSES: dict[str, AnalysisFn] = {
    "structure": run_structure,
    "viscosity": run_viscosity,
}
