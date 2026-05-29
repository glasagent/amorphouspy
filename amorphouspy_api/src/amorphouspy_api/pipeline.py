"""Pipeline orchestration for amorphouspy API.

Registers step functions that adapt pydantic models to core library calls,
and provides ``submit_pipeline`` to wire them into an executorlib DAG.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from concurrent.futures import Future

    from executorlib.executor.base import BaseExecutor
    from pydantic import BaseModel

    from amorphouspy_api.models import (
        CTEFluctuations,
        CTETemperatureScan,
        ElasticAnalysis,
        JobSubmission,
        StructureAnalysis,
        ViscosityAnalysis,
    )

logger = logging.getLogger(__name__)

AnalysisFn = Callable[..., dict]


# ---------------------------------------------------------------------------
# Step wrapper functions
# ---------------------------------------------------------------------------


def _generate_structure(submission: JobSubmission, config: BaseModel, result: dict) -> dict:
    """Generate initial structure and potential from composition."""
    from amorphouspy.pipelines.meltquench import generate_structure

    return generate_structure(
        composition=submission.composition.root,
        n_atoms=submission.simulation.n_atoms,
        potential_type=submission.potential,
        density=submission.simulation.target_density,
        structure_seed=submission.simulation.structure_seed,
        electrostatics_config=submission.electrostatics.to_electrostatics_config(),
    )


def _run_melt_quench(submission: JobSubmission, config: BaseModel, result: dict) -> dict:
    """Run the LAMMPS melt-quench simulation."""
    from amorphouspy.pipelines.meltquench import run_melt_quench

    from amorphouspy_api.executor import get_lammps_server_kwargs

    mq_result = run_melt_quench(
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


def _run_structural_analysis(submission: JobSubmission, config: StructureAnalysis, result: dict) -> dict:
    """Structural analysis (RDF, coordination, bond angles) on the quenched glass."""
    from amorphouspy.pipelines.structural import run_structural_analysis

    mq = result["melt_quench"]
    mean_data, _sem_data, n_frames = run_structural_analysis(
        final_structure=mq["final_structure"],
        simulation_history=mq.get("simulation_history"),
    )
    result_dict = mean_data.model_dump()
    result_dict["n_averaging_frames"] = n_frames
    return result_dict


def _run_viscosity(submission: JobSubmission, config: ViscosityAnalysis, result: dict) -> dict:
    """Multi-temperature viscosity analysis on the quenched glass."""
    from amorphouspy.pipelines.viscosity import run_viscosity_workflow

    from amorphouspy_api.executor import get_lammps_server_kwargs

    return run_viscosity_workflow(
        structure=result["melt_quench"]["final_structure"],
        potential=result["structure_generation"]["potential"],
        temperatures=config.temperatures,
        heating_rate=int(submission.simulation.quench_rate * 100),
        cooling_rate=int(submission.simulation.quench_rate),
        timestep=config.timestep,
        n_timesteps=config.n_timesteps,
        n_print=config.n_print,
        max_lag=config.max_lag,
        server_kwargs=get_lammps_server_kwargs(),
    )


def _run_cte(submission: JobSubmission, config: CTEFluctuations | CTETemperatureScan, result: dict) -> dict:
    """CTE analysis via fluctuations or temperature scan."""
    from amorphouspy.properties.cte import cte_from_fluctuations_simulation, temperature_scan_simulation

    from amorphouspy_api.executor import get_lammps_server_kwargs
    from amorphouspy_api.models import CTEFluctuations, CTETemperatureScan

    potential = result["structure_generation"]["potential"]
    structure = result["melt_quench"]["final_structure"]
    resource_dict = get_lammps_server_kwargs()

    if isinstance(config, CTEFluctuations):
        cte_result = cte_from_fluctuations_simulation(
            structure=structure,
            potential=potential,
            temperature=config.temperature,
            pressure=config.pressure,
            timestep=config.timestep,
            equilibration_steps=config.equilibration_steps,
            production_steps=config.production_steps,
            min_production_runs=config.min_production_runs,
            max_production_runs=config.max_production_runs,
            CTE_uncertainty_criterion=config.cte_uncertainty_criterion,
            server_kwargs=resource_dict,
        )
        cte_result["metadata"] = {
            "temperature": config.temperature,
            "production_steps": config.production_steps,
            "timestep": config.timestep,
        }
        return cte_result

    assert isinstance(config, CTETemperatureScan)
    cte_result = temperature_scan_simulation(
        structure=structure,
        potential=potential,
        temperature=config.temperatures,
        pressure=config.pressure,
        timestep=config.timestep,
        equilibration_steps=config.equilibration_steps,
        production_steps=config.production_steps,
        server_kwargs=resource_dict,
    )
    cte_result["metadata"] = {
        "temperatures": config.temperatures,
        "production_steps": config.production_steps,
        "timestep": config.timestep,
    }
    return cte_result


def _run_elastic(submission: JobSubmission, config: ElasticAnalysis, result: dict) -> dict:
    """Elastic moduli analysis on the quenched glass."""
    from amorphouspy.properties.elastic import elastic_simulation

    from amorphouspy_api.executor import get_lammps_server_kwargs

    raw = elastic_simulation(
        structure=result["melt_quench"]["final_structure"],
        potential=result["structure_generation"]["potential"],
        temperature_sim=config.temperature,
        pressure=config.pressure,
        timestep=config.timestep,
        equilibration_steps=config.equilibration_steps,
        production_steps=config.production_steps,
        n_print=config.n_print,
        strain=config.strain,
        server_kwargs=get_lammps_server_kwargs(),
    )

    cij = raw.get("Cij")
    if cij is not None and hasattr(cij, "tolist"):
        cij = cij.tolist()

    return {"Cij": cij, "moduli": raw.get("moduli", {})}


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEPS: dict[str, AnalysisFn] = {
    "structure_generation": _generate_structure,
    "melt_quench": _run_melt_quench,
    "structure_characterization": _run_structural_analysis,
    "viscosity": _run_viscosity,
    "cte": _run_cte,
    "elastic": _run_elastic,
}

BASE_STEPS = {"structure_generation", "melt_quench"}
ANALYSES: dict[str, AnalysisFn] = {k: v for k, v in STEPS.items() if k not in BASE_STEPS}

__all__ = ["ANALYSES", "BASE_STEPS", "STEPS", "submit_pipeline"]


# ---------------------------------------------------------------------------
# DAG orchestration
# ---------------------------------------------------------------------------


def _accumulate_step(
    step_name: str,
    step_fn: AnalysisFn,
    submission: JobSubmission,
    config: BaseModel | None,
    accumulated: dict,
) -> dict:
    """Run one pipeline step and merge its output into the accumulated dict."""
    step_result = step_fn(submission, config, accumulated)
    return {**accumulated, step_name: step_result}


def _run_analysis(
    step_name: str,
    step_fn: AnalysisFn,
    submission: JobSubmission,
    config: BaseModel,
    base_result: dict,
) -> dict:
    """Run a single analysis step. Returns ``{step_name: result}``."""
    step_result = step_fn(submission, config, base_result)
    return {step_name: step_result}


def _merge_results(base_result: dict, **analysis_results: dict) -> dict:
    """Merge the base pipeline result with individual analysis outputs."""
    merged = dict(base_result)
    for result_dict in analysis_results.values():
        merged.update(result_dict)
    return merged


def submit_pipeline(
    executor: BaseExecutor,
    submission: JobSubmission,
    cache_key: str | None = None,
) -> Future:
    """Submit all pipeline steps as executor futures.

    Base steps (structure_generation, melt_quench) run sequentially.
    Requested analyses then fan out **in parallel** from the base result.
    A final merge step collects everything under the bare *cache_key*.
    """
    from amorphouspy_api.executor import _is_slurm, get_base_resource_dict, get_lammps_resource_dict

    base_resource_dict = get_base_resource_dict()
    lammps_resource_dict = get_lammps_resource_dict()

    # Steps that run LAMMPS simulations and need multi-core SBATCH allocation.
    LAMMPS_STEPS = {"melt_quench", "cte", "viscosity", "elastic"}

    # --- Base steps: sequential chain ---
    future = None
    for name in ("structure_generation", "melt_quench"):
        rd = lammps_resource_dict if name in LAMMPS_STEPS else base_resource_dict
        resource_dict = dict(rd)
        if _is_slurm():
            resource_dict["job_name"] = name
        if cache_key is not None:
            resource_dict["cache_key"] = f"{cache_key}_{name}"
        future = executor.submit(
            _accumulate_step,
            resource_dict=resource_dict,
            step_name=name,
            step_fn=STEPS[name],
            submission=submission,
            config=None,
            accumulated=future if future is not None else {},
        )

    base_future = future  # contains structure_generation + melt_quench

    # --- Analysis steps: fan-out in parallel from base_future ---
    analysis_configs = {a.type: a for a in submission.analyses}
    analysis_futures: dict[str, Future] = {}
    for name, config in analysis_configs.items():
        if name in ANALYSES:
            rd = lammps_resource_dict if name in LAMMPS_STEPS else base_resource_dict
            resource_dict = dict(rd)
            if _is_slurm():
                resource_dict["job_name"] = name
            if cache_key is not None:
                resource_dict["cache_key"] = f"{cache_key}_{name}"
            analysis_futures[name] = executor.submit(
                _run_analysis,
                resource_dict=resource_dict,
                step_name=name,
                step_fn=ANALYSES[name],
                submission=submission,
                config=config,
                base_result=base_future,
            )

    # --- Merge step: collects base + all analysis results ---
    merge_resource: dict[str, Any] = dict(base_resource_dict)
    if _is_slurm():
        merge_resource["job_name"] = "merge_results"
    if cache_key is not None:
        merge_resource["cache_key"] = cache_key
    merge_kwargs: dict[str, dict | Future | None] = {"base_result": base_future}
    merge_kwargs.update(analysis_futures)

    return executor.submit(_merge_results, resource_dict=merge_resource, **merge_kwargs)
