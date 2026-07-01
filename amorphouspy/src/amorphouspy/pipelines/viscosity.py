"""Multi-temperature viscosity pipeline.

Cools a quenched glass to each target temperature via melt-quench, then
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
    from concurrent.futures import Executor, Future

    # executorlib extends Executor.submit() with resource_dict and
    # future-as-argument DAG resolution.  We type the DAG-building function
    # with Any to avoid false positives from the stdlib stubs.
    import pandas as pd
    from ase import Atoms

    type DagExecutor = Any
from amorphouspy.properties.viscosity import get_viscosity, viscosity_simulation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers submitted to the executor
# ---------------------------------------------------------------------------


def _cool_and_run_viscosity(
    structure: Atoms,
    potential: pd.DataFrame,
    temp_high: float,
    temp_low: float,
    heating_rate: float,
    cooling_rate: float,
    timestep: float,
    n_timesteps: int,
    n_print: int,
    max_lag: int | None,
    server_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Cool to *temp_low* via melt-quench, then run a viscosity simulation.

    This function is submitted to the executor as a single unit of work
    so that the melt-quench and the production run share the same worker.
    """
    mq_result = melt_quench_simulation(
        structure=structure,
        potential=potential,
        temperature_high=float(temp_high),
        temperature_low=float(temp_low),
        timestep=1.0,
        heating_rate=float(heating_rate),
        cooling_rate=float(cooling_rate),
        n_print=1000,
        langevin=False,
        server_kwargs=server_kwargs,
    )
    cooled_structure = mq_result["structure"]

    visc_result = viscosity_simulation(
        structure=cooled_structure,
        potential=potential,
        temperature_sim=float(temp_low),
        timestep=float(timestep),
        initial_production_steps=int(n_timesteps),
        n_print=int(n_print),
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


def _run_single_viscosity(
    base_result: dict,
    temp_high: float,
    temperature: float,
    heating_rate: float,
    cooling_rate: float,
    timestep: float,
    n_timesteps: int,
    n_print: int,
    max_lag: int | None,
    server_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Extract structure from *base_result*, cool, and run viscosity."""
    return _cool_and_run_viscosity(
        structure=base_result["melt_quench"]["final_structure"],
        potential=base_result["structure_generation"]["potential"],
        temp_high=temp_high,
        temp_low=temperature,
        heating_rate=heating_rate,
        cooling_rate=cooling_rate,
        timestep=timestep,
        n_timesteps=n_timesteps,
        n_print=n_print,
        max_lag=max_lag,
        server_kwargs=server_kwargs,
    )


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
    temp_high: float,
    heating_rate: float,
    cooling_rate: float,
    timestep: float,
    n_timesteps: int,
    n_print: int,
    max_lag: int | None,
    server_kwargs: dict[str, Any],
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
        base_future: Future resolving to the base pipeline result dict
            (must contain ``melt_quench`` and ``structure_generation`` keys).
        temperatures: Target temperatures (K).
        temp_high: Starting melt temperature (K) for the cooling step.
        heating_rate: Heating rate in K/ps.
        cooling_rate: Cooling rate in K/ps.
        timestep: MD timestep in fs.
        n_timesteps: Production run length (steps).
        n_print: Thermodynamic output frequency.
        max_lag: Maximum correlation lag (steps) for Green-Kubo.
        server_kwargs: LAMMPS server configuration.
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
            temperature=temp,
            heating_rate=heating_rate,
            cooling_rate=cooling_rate,
            timestep=timestep,
            n_timesteps=n_timesteps,
            n_print=n_print,
            max_lag=max_lag,
            server_kwargs=server_kwargs,
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


def run_viscosity_workflow(
    executor: Executor,
    structure: Atoms,
    potential: pd.DataFrame,
    temperatures: list[float],
    heating_rate: float,
    cooling_rate: float,
    timestep: float = 1.0,
    n_timesteps: int = 10_000_000,
    n_print: int = 1,
    max_lag: int | None = 1_000_000,
    server_kwargs: dict[str, Any] | None = None,
    resource_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run viscosity analysis at multiple temperatures using an executor.

    Each temperature is submitted as an independent task to *executor*.
    Every task first cools the starting structure to the target temperature
    via a melt-quench, then performs a Green-Kubo viscosity production run.
    Because each task starts from the same *structure* and cools
    independently, the tasks can run in parallel.

    Args:
        executor: A ``concurrent.futures.Executor`` or executorlib executor
            used to submit per-temperature tasks.
        structure: Quenched ASE Atoms from the melt-quench pipeline.
        potential: LAMMPS potential object.
        temperatures: Target temperatures (K) for viscosity runs.
        heating_rate: Heating rate in K/ps.
        cooling_rate: Cooling rate in K/ps.
        timestep: MD timestep in fs.
        n_timesteps: Number of MD steps per viscosity production run.
        n_print: Thermodynamic output frequency.
        max_lag: Maximum correlation lag (steps) for Green-Kubo.
        server_kwargs: LAMMPS server/resource configuration.
        resource_dict: Optional resource dict forwarded to executorlib
            executors (e.g. ``{"cores": 8}``).  Ignored for stdlib executors.

    Returns:
        Result dict with ``temperatures``, ``viscosities``, ``max_lag``,
        ``simulation_steps``, ``lag_times_ps``, and ``viscosity_integral``.
    """
    if server_kwargs is None:
        server_kwargs = {}
    if resource_dict is None:
        resource_dict = {}

    sorted_temps = sorted(temperatures, reverse=True)
    logger.info("Submitting viscosity tasks for temperatures: %s", sorted_temps)

    # Submit all temperatures in parallel — each cools independently
    # from the starting structure down to its target temperature.
    futures = {}
    for temp in sorted_temps:
        submit_kwargs: dict[str, Any] = {
            "structure": structure,
            "potential": potential,
            "temp_high": 5000.0,
            "temp_low": temp,
            "heating_rate": heating_rate,
            "cooling_rate": cooling_rate,
            "timestep": timestep,
            "n_timesteps": n_timesteps,
            "n_print": n_print,
            "max_lag": max_lag,
            "server_kwargs": server_kwargs,
        }
        if resource_dict:
            submit_kwargs["resource_dict"] = resource_dict
        futures[temp] = executor.submit(_cool_and_run_viscosity, **submit_kwargs)

    # Collect results in temperature order
    viscosities: list[float] = []
    all_max_lags: list[float] = []
    sim_steps: list[int] = []
    lag_times_ps: list[list[float]] = []
    viscosity_integral: list[list[float]] = []

    for temp in sorted_temps:
        result = futures[temp].result()
        logger.info("Viscosity at %.1f K: %.3e Pa·s", temp, result["viscosity"])
        viscosities.append(result["viscosity"])
        all_max_lags.append(result["max_lag"])
        sim_steps.append(result["simulation_steps"])
        lag_times_ps.append(result["lag_times_ps"])
        viscosity_integral.append(result["viscosity_integral"])

    return {
        "temperatures": sorted_temps,
        "viscosities": viscosities,
        "max_lag": all_max_lags,
        "simulation_steps": sim_steps,
        "lag_times_ps": lag_times_ps,
        "viscosity_integral": viscosity_integral,
    }
