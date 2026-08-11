"""Pipeline orchestration for amorphouspy API.

Registers step functions that adapt pydantic models to core library calls,
and provides ``submit_pipeline`` to wire them into an executorlib DAG.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
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

from amorphouspy_api.executor import get_lammps_server_kwargs

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
        use_three_body=submission.potential_config.use_three_body,
    )


def _run_melt_quench(submission: JobSubmission, config: BaseModel, result: dict) -> dict:
    """Run the LAMMPS melt-quench simulation."""
    from amorphouspy.pipelines.meltquench import run_melt_quench

    mq_result = run_melt_quench(
        structure=result["structure_generation"]["structure"],
        potential=result["structure_generation"]["potential"],
        potential_type=submission.potential,
        cooling_rate=int(submission.simulation.quench_rate),
        timestep=submission.simulation.timestep,
        temperature_high=submission.simulation.melt_temperature,
        temperature_low=300.0,
        equilibration_steps=submission.simulation.equilibration_steps,
        n_dump=submission.simulation.n_dump,
        n_print_thermo=submission.simulation.n_print_thermo,
        server_kwargs=get_lammps_server_kwargs(
            submission.potential, submission.simulation.n_atoms, submission.simulation.cores
        ),
    )
    mq_result["composition"] = submission.composition.root
    return mq_result


def _run_structural_analysis(submission: JobSubmission, config: StructureAnalysis, result: dict) -> dict:
    """Structural analysis (RDF, coordination, bond angles) on the quenched glass.

    This is a separate post-melt-quench step. When ``n_averaging_frames > 1``,
    it runs an additional NVT sampling simulation on the final quenched
    structure before computing averaged structural observables.
    """
    from amorphouspy.properties.structural.all import run_structural_analysis

    from amorphouspy_api.models import StructuralAnalysisTrajectoryStorageMode

    mq = result["melt_quench"]
    potential = result["structure_generation"]["potential"]

    from amorphouspy_api.database import _apply_trajectory_storage_mode

    # Calculate n_bins from rdf_cutoff and bin_width
    n_bins = int(config.rdf_cutoff / config.bin_width)

    mean_data, _sem_data, n_frames, sampling_history = run_structural_analysis(
        final_structure=mq["final_structure"],
        potential=potential,
        timestep=submission.simulation.timestep,
        n_averaging_frames=config.n_averaging_frames,
        temperature=300.0,
        r_max=config.rdf_cutoff,
        n_bins=n_bins,
        server_kwargs=get_lammps_server_kwargs(
            submission.potential, submission.simulation.n_atoms, submission.simulation.cores
        ),
    )
    result_dict = mean_data.model_dump()
    result_dict["n_averaging_frames"] = n_frames
    if (
        sampling_history is not None
        and submission.simulation.structural_analysis_trajectory_storage_mode
        != StructuralAnalysisTrajectoryStorageMode.NO_DUMP_DATA
    ):
        result_dict["sampling_history"] = _apply_trajectory_storage_mode(
            sampling_history,
            trajectory_storage_mode=submission.simulation.structural_analysis_trajectory_storage_mode,
        )
    return result_dict


def _submit_viscosity(
    executor: BaseExecutor,
    base_future: Future,
    submission: JobSubmission,
    config: ViscosityAnalysis,
    lammps_resource_dict: dict[str, Any],
    base_resource_dict: dict[str, Any],
    *,
    is_slurm: bool,
    cache_key: str | None,
) -> Future:
    """Thin adapter: extract plain args and delegate to the core pipeline."""
    from amorphouspy.pipelines.viscosity import submit_viscosity_workflow

    from amorphouspy_api.executor import get_lammps_server_kwargs

    melt_temp = submission.simulation.melt_temperature
    return submit_viscosity_workflow(
        executor=executor,
        base_future=base_future,
        temperatures=config.temperatures,
        temp_high=float(melt_temp) if melt_temp is not None else None,
        cooling_rate=int(submission.simulation.quench_rate),
        timestep=config.timestep,
        n_timesteps=config.n_timesteps,
        n_dump=config.n_dump,
        n_print_thermo=config.n_print_thermo,
        max_lag=config.max_lag,
        server_kwargs=get_lammps_server_kwargs(
            submission.potential, submission.simulation.n_atoms, submission.simulation.cores
        ),
        equilibration_steps=submission.simulation.equilibration_steps,
        lammps_resource_dict=lammps_resource_dict,
        base_resource_dict=base_resource_dict,
        is_slurm=is_slurm,
        cache_key=cache_key,
    )


def _run_cte(
    submission: JobSubmission,
    config: CTEFluctuations | CTETemperatureScan,
    result: dict,
) -> dict:
    """CTE analysis via fluctuations or temperature scan."""
    from amorphouspy.properties.cte import cte_from_fluctuations_simulation, temperature_scan_simulation

    from amorphouspy_api.models import CTEFluctuations, CTETemperatureScan

    potential = result["structure_generation"]["potential"]
    structure = result["melt_quench"]["final_structure"]
    resource_dict = get_lammps_server_kwargs(
        submission.potential, submission.simulation.n_atoms, submission.simulation.cores
    )

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

    raw = elastic_simulation(
        structure=result["melt_quench"]["final_structure"],
        potential=result["structure_generation"]["potential"],
        temperature_sim=config.temperature,
        pressure=config.pressure,
        timestep=config.timestep,
        equilibration_steps=config.equilibration_steps,
        production_steps=config.production_steps,
        n_dump=config.n_dump,
        n_print_thermo=config.n_print_thermo,
        strain=config.strain,
        server_kwargs=get_lammps_server_kwargs(
            submission.potential, submission.simulation.n_atoms, submission.simulation.cores
        ),
    )

    cij = raw.get("Cij")
    if cij is not None and hasattr(cij, "tolist"):
        cij = cij.tolist()

    return {"Cij": cij, "moduli": raw.get("moduli", {})}


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepSpec:
    """Metadata describing how a single pipeline step is scheduled.

    Attributes:
        fn: The step function. For ``submits=False`` steps this is an
            ``AnalysisFn`` run inside an executor task; for ``submits=True``
            steps it is a self-submitting function returning a ``Future``.
        is_base: Part of the sequential base chain (structure_generation ->
            melt_quench) that every analysis builds on.
        submits: Calls ``executor.submit`` itself and returns a ``Future``,
            rather than being wrapped in ``_run_analysis``.
        lammps: Runs a LAMMPS simulation and needs the multi-core resource dict.
        depends_on: Name of the step whose future this step consumes. ``None``
            marks the start of the base chain (structure_generation). Analyses
            default to ``"melt_quench"`` (the full base); viscosity depends only
            on ``"structure_generation"`` so it runs in parallel with the main
            melt-quench.
        display_name: Human-readable label shown in the UI (e.g. timing table).
    """

    fn: Callable[..., Any]
    display_name: str
    is_base: bool = False
    submits: bool = False
    lammps: bool = False
    depends_on: str | None = None


# Single source of truth for every pipeline step, in execution order.
REGISTRY: dict[str, StepSpec] = {
    "structure_generation": StepSpec(_generate_structure, display_name="Structure Generation", is_base=True),
    "melt_quench": StepSpec(
        _run_melt_quench, display_name="Melt-Quench", is_base=True, lammps=True, depends_on="structure_generation"
    ),
    "structure_characterization": StepSpec(
        _run_structural_analysis, display_name="Structural Analysis", depends_on="melt_quench"
    ),
    "cte": StepSpec(_run_cte, display_name="CTE", lammps=True, depends_on="melt_quench"),
    "elastic": StepSpec(_run_elastic, display_name="Elastic Properties", lammps=True, depends_on="melt_quench"),
    "viscosity": StepSpec(
        _submit_viscosity, display_name="Viscosity", submits=True, lammps=True, depends_on="structure_generation"
    ),
}

# Derived views over REGISTRY, kept for external importers and tests.
# ``STEPS`` holds the base steps plus the simple (non-self-submitting) analyses.
STEPS: dict[str, AnalysisFn] = {name: spec.fn for name, spec in REGISTRY.items() if not spec.submits}
# Human-readable label per step, used by the UI (e.g. the timing table).
DISPLAY_NAMES: dict[str, str] = {name: spec.display_name for name, spec in REGISTRY.items()}
# Ordered so callers can rely on the base chain sequence (structure_generation
# before melt_quench); still supports ``in`` membership tests.
BASE_STEPS: tuple[str, ...] = tuple(name for name, spec in REGISTRY.items() if spec.is_base)
ANALYSES: dict[str, AnalysisFn] = {
    name: spec.fn for name, spec in REGISTRY.items() if not spec.is_base and not spec.submits
}
# Analyses that build their own sub-DAG instead of going through _run_analysis.
_SUBMITTERS: dict[str, Callable[..., Future]] = {name: spec.fn for name, spec in REGISTRY.items() if spec.submits}
# All known analysis names (simple + DAG-based), used for result lookups and
# progress tracking.
ANALYSIS_NAMES: frozenset[str] = frozenset(name for name, spec in REGISTRY.items() if not spec.is_base)

__all__ = [
    "ANALYSES",
    "ANALYSIS_NAMES",
    "BASE_STEPS",
    "DISPLAY_NAMES",
    "REGISTRY",
    "STEPS",
    "StepSpec",
    "submit_pipeline",
]


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


def _build_resource_dict(
    base_rd: dict[str, Any],
    step_name: str,
    *,
    is_slurm: bool,
    cache_key: str | None,
) -> dict[str, Any]:
    """Build a resource dict with optional job-name and cache-key."""
    rd = dict(base_rd)
    if is_slurm:
        rd["job_name"] = step_name
    if cache_key is not None:
        rd["cache_key"] = f"{cache_key}_{step_name}"
    return rd


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
    lammps_resource_dict = get_lammps_resource_dict(
        submission.potential, submission.simulation.n_atoms, submission.simulation.cores
    )
    slurm = _is_slurm()

    def _resource_for(spec: StepSpec) -> dict[str, Any]:
        return lammps_resource_dict if spec.lammps else base_resource_dict

    # Futures keyed by step name, so any step can look up its dependency's
    # future by name (see ``StepSpec.depends_on``).
    futures: dict[str, Future] = {}

    def _submit_base_step(name: str) -> Future:
        """Submit one base step, chaining on its ``depends_on`` predecessor."""
        spec = REGISTRY[name]
        accumulated = futures[spec.depends_on] if spec.depends_on else {}
        resource_dict = _build_resource_dict(_resource_for(spec), name, is_slurm=slurm, cache_key=cache_key)
        return executor.submit(
            _accumulate_step,
            resource_dict=resource_dict,
            step_name=name,
            step_fn=spec.fn,
            submission=submission,
            config=None,
            accumulated=accumulated,
        )

    # --- Base steps: explicit sequential chain ---
    futures["structure_generation"] = _submit_base_step("structure_generation")
    futures["melt_quench"] = _submit_base_step("melt_quench")
    base_future = futures["melt_quench"]  # base = structure_gen + melt_quench

    # --- Analysis steps: fan-out in parallel ---
    analysis_configs = {a.type: a for a in submission.analyses}
    analysis_futures: dict[str, Future] = {}
    for name, config in analysis_configs.items():
        spec = REGISTRY.get(name)
        if spec is None or spec.is_base:
            continue
        # Each analysis consumes the future of the step named by ``depends_on``.
        # Steps that depend on structure_generation run in parallel with the
        # main melt-quench; the rest wait for the full base chain.
        dep_future = futures[spec.depends_on] if spec.depends_on else base_future
        if spec.submits:
            analysis_futures[name] = spec.fn(
                executor=executor,
                base_future=dep_future,
                submission=submission,
                config=config,
                lammps_resource_dict=lammps_resource_dict,
                base_resource_dict=base_resource_dict,
                is_slurm=slurm,
                cache_key=cache_key,
            )
        else:
            resource_dict = _build_resource_dict(_resource_for(spec), name, is_slurm=slurm, cache_key=cache_key)
            analysis_futures[name] = executor.submit(
                _run_analysis,
                resource_dict=resource_dict,
                step_name=name,
                step_fn=spec.fn,
                submission=submission,
                config=config,
                base_result=dep_future,
            )

    # --- Merge step: collects base + all analysis results ---
    merge_resource: dict[str, Any] = dict(base_resource_dict)
    if slurm:
        merge_resource["job_name"] = "merge_results"
    if cache_key is not None:
        merge_resource["cache_key"] = cache_key
    merge_kwargs: dict[str, dict | Future | None] = {"base_result": base_future}
    merge_kwargs.update(analysis_futures)

    return executor.submit(_merge_results, resource_dict=merge_resource, **merge_kwargs)
