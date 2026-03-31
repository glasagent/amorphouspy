"""Workflow functions for amorphouspy API."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .analyses import run_cte, run_elastic, run_structural_analysis, run_viscosity
from .meltquench import generate_structure, run_melt_quench

AnalysisFn = Callable[["JobSubmission", "BaseModel", dict], dict]

STEPS: dict[str, AnalysisFn] = {
    "structure_generation": generate_structure,
    "melt_quench": run_melt_quench,
    "structure": run_structural_analysis,
    "viscosity": run_viscosity,
    "cte": run_cte,
    "elastic": run_elastic,
}

BASE_STEPS = {"structure_generation", "melt_quench"}
ANALYSES: dict[str, AnalysisFn] = {k: v for k, v in STEPS.items() if k not in BASE_STEPS}

if TYPE_CHECKING:
    from concurrent.futures import Future

    from executorlib.executor.base import BaseExecutor
    from pydantic import BaseModel

    from amorphouspy_api.models import JobSubmission

__all__ = ["ANALYSES", "BASE_STEPS", "STEPS", "generate_structure", "run_melt_quench", "submit_pipeline"]


def _accumulate_step(step_name: str, step_fn, submission, config, accumulated: dict) -> dict:
    """Run one pipeline step and merge its output into the accumulated dict.

    This function is submitted to executorlib; all arguments must be picklable.
    """
    step_result = step_fn(submission, config, accumulated)
    return {**accumulated, step_name: step_result}


def _run_analysis(step_name: str, step_fn, submission, config, base_result: dict) -> dict:
    """Run a single analysis step. Returns ``{step_name: result}``.

    Unlike ``_accumulate_step`` this does **not** carry forward the full
    accumulated dict — each analysis only receives the base pipeline result
    (structure_generation + melt_quench) and works independently.
    """
    step_result = step_fn(submission, config, base_result)
    return {step_name: step_result}


def _merge_results(base_result: dict, **analysis_results: dict) -> dict:
    """Merge the base pipeline result with individual analysis outputs."""
    merged = dict(base_result)
    for result_dict in analysis_results.values():
        merged.update(result_dict)
    return merged


def _make_resource_dict(
    base: dict[str, Any],
    job_name: str,
    cache_key: str | None,
    step_name: str,
    is_slurm: bool,  # noqa: FBT001
) -> dict[str, Any]:
    """Build a resource_dict for a single pipeline step."""
    rd = dict(base)
    if is_slurm:
        rd["job_name"] = job_name
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

    Each intermediate step gets ``{cache_key}_{step_name}`` so individual
    progress can be queried via ``get_future_from_cache``.
    """
    from amorphouspy_api.executor import _is_slurm, get_base_resource_dict, get_lammps_resource_dict

    base_resource_dict = get_base_resource_dict()
    lammps_resource_dict = get_lammps_resource_dict()
    slurm = _is_slurm()

    # Steps that run LAMMPS simulations and need multi-core SBATCH allocation.
    LAMMPS_STEPS = {"melt_quench", "cte", "viscosity", "elastic"}

    def _rd_for(name: str) -> dict[str, Any]:
        base = lammps_resource_dict if name in LAMMPS_STEPS else base_resource_dict
        return _make_resource_dict(base, name, cache_key, name, slurm)

    # --- Base steps: sequential chain ---
    future = None
    for name in ("structure_generation", "melt_quench"):
        future = executor.submit(
            _accumulate_step,
            resource_dict=_rd_for(name),
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
        if name == "viscosity":
            from .analyses.viscosity import submit_viscosity_substeps

            analysis_futures[name] = submit_viscosity_substeps(
                executor=executor,
                base_future=base_future,
                submission=submission,
                config=config,
                cache_key=cache_key,
                lammps_rd=lammps_resource_dict,
                base_rd=base_resource_dict,
                is_slurm=slurm,
            )
        elif name in ANALYSES:
            analysis_futures[name] = executor.submit(
                _run_analysis,
                resource_dict=_rd_for(name),
                step_name=name,
                step_fn=ANALYSES[name],
                submission=submission,
                config=config,
                base_result=base_future,
            )

    # --- Merge step: collects base + all analysis results ---
    merge_resource = _make_resource_dict(base_resource_dict, "merge_results", cache_key, "", slurm)
    if cache_key is not None:
        merge_resource["cache_key"] = cache_key  # bare key, no step suffix
    merge_kwargs: dict[str, dict | Future] = {"base_result": base_future}
    merge_kwargs.update(analysis_futures)

    return executor.submit(_merge_results, resource_dict=merge_resource, **merge_kwargs)
