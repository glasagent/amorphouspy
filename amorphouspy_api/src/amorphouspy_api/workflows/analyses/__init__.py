"""Analysis step implementations (structure, viscosity, cte)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

    from amorphouspy_api.models import JobSubmission, ViscosityAnalysis


def run_structural_analysis(submission: JobSubmission, config: BaseModel, result: dict) -> dict:
    """Structural analysis (RDF, coordination, bond angles) on the quenched glass."""
    from amorphouspy.workflows.structural_analysis import analyze_structure

    data = analyze_structure(atoms=result["melt_quench"]["final_structure"])
    return data.model_dump() if hasattr(data, "model_dump") else data


def run_viscosity(submission: JobSubmission, config: BaseModel, result: dict) -> dict:
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
        structure=result["melt_quench"]["final_structure"],
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


def run_cte(submission: JobSubmission, config: BaseModel, result: dict) -> dict:
    """CTE analysis via fluctuations or temperature scan."""
    from amorphouspy import generate_potential, get_structure_dict

    from amorphouspy_api.executor import get_lammps_resource_dict
    from amorphouspy_api.models import CTEFluctuations

    atoms_dict = get_structure_dict(
        composition=submission.composition.root,
        target_atoms=submission.simulation.n_atoms,
    )
    potential = generate_potential(
        atoms_dict=atoms_dict,
        potential_type=submission.potential.value,
    )

    resource_dict = get_lammps_resource_dict()

    if isinstance(config, CTEFluctuations):
        from amorphouspy_api.workflows.analyses.cte import run_cte_fluctuations

        return run_cte_fluctuations(
            structure=result["melt_quench"]["final_structure"],
            potential=potential,
            temperature=config.temperature,
            pressure=config.pressure,
            timestep=config.timestep,
            equilibration_steps=config.equilibration_steps,
            production_steps=config.production_steps,
            min_production_runs=config.min_production_runs,
            max_production_runs=config.max_production_runs,
            cte_uncertainty_criterion=config.cte_uncertainty_criterion,
            lammps_resource_dict=resource_dict,
        )

    from amorphouspy_api.workflows.analyses.cte import run_cte_temperature_scan

    return run_cte_temperature_scan(
        structure=result["melt_quench"]["final_structure"],
        potential=potential,
        temperatures=config.temperatures,
        pressure=config.pressure,
        timestep=config.timestep,
        equilibration_steps=config.equilibration_steps,
        production_steps=config.production_steps,
        lammps_resource_dict=resource_dict,
    )


__all__ = ["run_cte", "run_structural_analysis", "run_viscosity"]
