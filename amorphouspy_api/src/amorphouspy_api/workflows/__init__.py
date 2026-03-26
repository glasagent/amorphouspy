"""Workflow functions for amorphouspy API."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .analyses import run_cte, run_structural_analysis, run_viscosity
from .meltquench import generate_structure, run_melt_quench

AnalysisFn = Callable[["JobSubmission", "BaseModel", dict], dict]

STEPS: dict[str, AnalysisFn] = {
    "structure_generation": generate_structure,
    "melt_quench": run_melt_quench,
    "structure": run_structural_analysis,
    "viscosity": run_viscosity,
    "cte": run_cte,
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


def submit_pipeline(
    executor: BaseExecutor,
    submission: JobSubmission,
    cache_key: str | None = None,
) -> Future:
    """Submit all pipeline steps as a chain of executor futures.

    Base steps (structure_generation, melt_quench) always run.
    Requested analyses are appended to the chain.
    The *cache_key* is set on the final step so the complete result
    can be retrieved later via ``get_future_from_cache``.
    """
    # Build ordered list of (step_name, step_fn, config)
    steps: list[tuple] = [(name, STEPS[name], None) for name in ("structure_generation", "melt_quench")]

    analysis_configs = {a.type: a for a in submission.analyses}
    steps.extend((name, ANALYSES[name], config) for name, config in analysis_configs.items() if name in ANALYSES)

    # Chain futures — each step receives the accumulated result from the previous
    future = None
    for i, (name, fn, config) in enumerate(steps):
        kwargs = {
            "step_name": name,
            "step_fn": fn,
            "submission": submission,
            "config": config,
            "accumulated": future if future is not None else {},
        }
        resource_dict = {}
        if i == len(steps) - 1 and cache_key is not None:
            resource_dict["cache_key"] = cache_key

        future = executor.submit(_accumulate_step, resource_dict=resource_dict, **kwargs)

    return future
