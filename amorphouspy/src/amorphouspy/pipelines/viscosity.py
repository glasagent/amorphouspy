"""Multi-temperature viscosity pipeline.

Cools the initial structure to each target temperature via melt-quench, then
runs Green-Kubo viscosity production simulations in parallel using an
executor.  The public entry-point ``submit_viscosity_workflow`` builds an
executorlib-style DAG (one task per temperature + a collect step) and
returns the final ``Future``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from amorphouspy.atoms.shared import downsample_log
from amorphouspy.fabrication.meltquench import melt_quench_simulation

if TYPE_CHECKING:
    from concurrent.futures import Future

    # executorlib extends Executor.submit() with resource_dict and
    # future-as-argument DAG resolution.  We type the DAG-building function
    # with Any to avoid false positives from the stdlib stubs.
    type DagExecutor = Any
from amorphouspy.properties.viscosity import get_viscosity, viscosity_simulation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers submitted to the executor
# ---------------------------------------------------------------------------


def _run_single_viscosity(
    base_result: dict,
    temp_high: float | None,
    temp_low: float,
    heating_rate: float,
    cooling_rate: float,
    timestep: float,
    n_timesteps: int,
    n_dump: int |None,
    n_print_thermo: int | None = 1,
    max_lag: int | None,
    server_kwargs: dict[str, Any],
    equilibration_steps: int | None = None,
) -> dict[str, Any]:
    """Cool the initial structure to *temp_low*, then run a viscosity simulation.

    Submitted to the executor as a single unit of work so the melt-quench and
    the production run share the same worker.  Starts from the freshly
    generated (random) structure rather than the quenched glass, so viscosity
    tasks can run in parallel with the main melt-quench.

    When *temp_high* is ``None`` the melt-quench uses the potential-dependent
    default melt temperature.
    """
    structure = base_result["structure_generation"]["structure"]
    potential = base_result["structure_generation"]["potential"]

    mq_result = melt_quench_simulation(
        structure=structure,
        potential=potential,
        temperature_high=float(temp_high) if temp_high is not None else None,
        temperature_low=float(temp_low),
        timestep=1.0,
        heating_rate=float(heating_rate),
        cooling_rate=float(cooling_rate),
        n_print=1000,
        equilibration_steps=equilibration_steps,
        langevin=False,
        server_kwargs=server_kwargs,
    )
    cooled_structure = mq_result["structure"]

        logger.info("Running viscosity simulation at %.1f K", temp)
        visc_result = viscosity_simulation(
            structure=structure_current,
            potential=potential,
            temperature_sim=float(temp),
            timestep=float(timestep),
            initial_production_steps=int(n_timesteps),
            n_dump=int(n_dump) if n_dump is not None else None,
            n_print_thermo=int(n_print_thermo) if n_print_thermo is not None else 1,
            langevin=False,
            seed=12345,
            server_kwargs=server_kwargs,
        )

    visc_data = get_viscosity(visc_result, timestep=float(timestep), max_lag=max_lag)

    raw_lag = cast("list[float]", visc_data.get("lag_time_ps", []))
    raw_visc = cast("list[float]", visc_data.get("viscosity_integral", []))

    return {
        "temperature": float(temp_low),
        "viscosity": float(visc_data["viscosity"]),
        "max_lag": float(visc_data["max_lag"]),
        "simulation_steps": int(n_timesteps),
        "lag_times_ps": downsample_log(raw_lag),
        "viscosity_integral": downsample_log(raw_visc),
    }


def _collect_viscosity_results(temperatures: list[float], **temp_results: dict) -> dict:
    """Merge per-temperature viscosity results into the final dict."""
    results = [temp_results[f"result_{i}"] for i in range(len(temperatures))]
    return {
        "viscosity": {
            "temperatures": temperatures,
            "viscosities": [r["viscosity"] for r in results],
            "max_lag": [r["max_lag"] for r in results],
            "simulation_steps": [r["simulation_steps"] for r in results],
            "lag_times_ps": [r["lag_times_ps"] for r in results],
            "viscosity_integral": [r["viscosity_integral"] for r in results],
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def submit_viscosity_workflow(
    executor: DagExecutor,
    base_future: Future,
    temperatures: list[float],
    temp_high: float | None,
    heating_rate: float,
    cooling_rate: float,
    timestep: float,
    n_timesteps: int,
    n_print: int,
    max_lag: int | None,
    server_kwargs: dict[str, Any],
    equilibration_steps: int | None = None,
    lammps_resource_dict: dict[str, Any] | None = None,
    base_resource_dict: dict[str, Any] | None = None,
    *,
    is_slurm: bool = False,
    cache_key: str | None = None,
) -> Future:
    """Submit per-temperature viscosity tasks and a collection step.

    Builds an executorlib-compatible DAG: one ``_run_single_viscosity``
    task per temperature (each depending on *base_future*), then a
    ``_collect_viscosity_results`` merge task depending on all of them.

    Args:
        executor: Executor (concurrent.futures or executorlib) to submit to.
        base_future: Future resolving to the structure-generation result dict
            (must contain a ``structure_generation`` key with the initial
            structure and potential).  Viscosity tasks depend only on this,
            so they run in parallel with the main melt-quench.
        temperatures: Target temperatures (K).
        temp_high: Starting melt temperature (K) for the cooling step.
            ``None`` uses the potential-dependent default melt temperature.
        heating_rate: Heating rate in K/ps.
        cooling_rate: Cooling rate in K/ps.
        timestep: MD timestep in fs.
        n_timesteps: Production run length (steps).
        n_print: Thermodynamic output frequency.
        max_lag: Maximum correlation lag (steps) for Green-Kubo.
        server_kwargs: LAMMPS server configuration.
        equilibration_steps: Override for the melt-quench equilibration stages;
            ``None`` uses each protocol's default.
        lammps_resource_dict: executorlib resource dict for LAMMPS tasks.
        base_resource_dict: executorlib resource dict for lightweight tasks.
        is_slurm: Whether to set ``job_name`` in resource dicts.
        cache_key: Optional cache key prefix.

    Returns:
        Future resolving to ``{"viscosity": {...}}``.
    """
    if lammps_resource_dict is None:
        lammps_resource_dict = {}
    if base_resource_dict is None:
        base_resource_dict = {}

    sorted_temps = sorted(temperatures, reverse=True)
    logger.info("Submitting viscosity tasks for temperatures: %s", sorted_temps)

    temp_futures: dict[str, Future] = {}
    for i, temp in enumerate(sorted_temps):
        rd: dict[str, Any] = dict(lammps_resource_dict)
        if is_slurm:
            rd["job_name"] = f"viscosity_{temp:.0f}K"
        if cache_key is not None:
            rd["cache_key"] = f"{cache_key}_viscosity_{temp:.0f}K"
        temp_futures[f"result_{i}"] = executor.submit(
            _run_single_viscosity,
            resource_dict=rd,
            base_result=base_future,
            temp_high=temp_high,
            temp_low=temp,
            heating_rate=heating_rate,
            cooling_rate=cooling_rate,
            timestep=timestep,
            n_timesteps=n_timesteps,
            n_print=n_print,
            max_lag=max_lag,
            server_kwargs=server_kwargs,
            equilibration_steps=equilibration_steps,
        )

    collect_rd: dict[str, Any] = dict(base_resource_dict)
    if is_slurm:
        collect_rd["job_name"] = "viscosity_collect"
    if cache_key is not None:
        collect_rd["cache_key"] = f"{cache_key}_viscosity"

    return executor.submit(
        _collect_viscosity_results,
        resource_dict=collect_rd,
        temperatures=sorted_temps,
        **temp_futures,
    )
