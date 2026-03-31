"""Viscosity workflow for glass simulation.

Runs Green-Kubo viscosity calculations at multiple temperatures starting
from an already-quenched glass structure.  The structure is sequentially
cooled from the highest to the lowest requested temperature, and at each
step a production MD run is performed followed by post-processing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from amorphouspy import melt_quench_simulation
from amorphouspy.workflows.viscosity import get_viscosity, viscosity_simulation

if TYPE_CHECKING:
    from concurrent.futures import Future

    from executorlib.executor.base import BaseExecutor
    from pydantic import BaseModel

    from amorphouspy_api.models import JobSubmission, ViscosityAnalysis

logger = logging.getLogger(__name__)


def run_viscosity(submission: JobSubmission, config: BaseModel, result: dict) -> dict:
    """Multi-temperature viscosity analysis on the quenched glass."""
    from amorphouspy_api.executor import get_lammps_server_kwargs

    cfg: ViscosityAnalysis = config  # type: ignore[assignment]

    return run_viscosity_workflow(
        structure=result["melt_quench"]["final_structure"],
        potential=result["structure_generation"]["potential"],
        temperatures=cfg.temperatures,
        heating_rate=int(submission.simulation.quench_rate * 100),
        cooling_rate=int(submission.simulation.quench_rate),
        timestep=cfg.timestep,
        n_timesteps=cfg.n_timesteps,
        n_print=cfg.n_print,
        max_lag=cfg.max_lag,
        lammps_resource_dict=get_lammps_server_kwargs(),
    )


def run_viscosity_workflow(
    structure,
    potential,
    temperatures: list[float],
    heating_rate: float,
    cooling_rate: float,
    timestep: float = 1.0,
    n_timesteps: int = 10_000_000,
    n_print: int = 1,
    max_lag: int | None = 1_000_000,
    lammps_resource_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run viscosity analysis at multiple temperatures.

    The workflow starts from the quenched structure produced by the
    melt-quench pipeline — it does **not** regenerate the structure or
    potential.

    Args:
        structure: Quenched ASE Atoms from the melt-quench workflow.
        potential: LAMMPS potential (from ``generate_potential``).
        temperatures: Target temperatures (K) for viscosity runs.
        heating_rate: Heating rate in K/ps (used when cooling between temps).
        cooling_rate: Cooling rate in K/ps.
        timestep: MD timestep in fs.
        n_timesteps: Number of MD steps per viscosity production run.
        n_print: Thermodynamic output frequency.
        max_lag: Maximum correlation lag (steps) for Green-Kubo.
        lammps_resource_dict: Resource dict for LAMMPS (e.g. {"cores": 4}).

    Returns:
        Result dict suitable for storing in ``result_data["viscosity"]``.
    """
    if lammps_resource_dict is None:
        lammps_resource_dict = {}

    # Sequential cooling: highest T → lowest T
    sorted_temps = sorted(temperatures, reverse=True)
    logger.info("Viscosity temperatures (high→low): %s", sorted_temps)

    viscosities: list[float] = []
    all_max_lags: list[float] = []
    sim_steps: list[int] = []
    lag_times_ps: list[list[float]] = []
    sacf_data: list[list[float]] = []
    viscosity_integral: list[list[float]] = []

    structure_current = structure

    for idx, temp in enumerate(sorted_temps):
        temp_high = 5000.0 if idx == 0 else sorted_temps[idx - 1]

        # Cool to this temperature via melt-quench
        logger.info("Cooling from %.1f K to %.1f K", temp_high, temp)
        mq_result = melt_quench_simulation(
            structure=structure_current,
            potential=potential,
            temperature_high=float(temp_high),
            temperature_low=float(temp),
            timestep=1.0,
            heating_rate=float(heating_rate),
            cooling_rate=float(cooling_rate),
            n_print=1000,
            langevin=False,
            server_kwargs=lammps_resource_dict,
        )
        structure_current = mq_result["structure"]
        logger.info("Cooled to %.1f K, %d atoms", temp, len(structure_current))

        # Run viscosity production simulation
        logger.info("Running viscosity simulation at %.1f K", temp)
        visc_result = viscosity_simulation(
            structure=structure_current,
            potential=potential,
            temperature_sim=float(temp),
            timestep=float(timestep),
            production_steps=int(n_timesteps),
            n_print=int(n_print),
            langevin=False,
            seed=12345,
            server_kwargs=lammps_resource_dict,
        )

        # Post-process: Green-Kubo analysis
        visc_data = get_viscosity(visc_result, timestep=float(timestep), max_lag=max_lag)
        logger.info("Viscosity at %.1f K: %.3e Pa·s", temp, visc_data["viscosity"])

        viscosities.append(float(visc_data["viscosity"]))
        all_max_lags.append(float(visc_data["max_lag"]))
        sim_steps.append(int(n_timesteps))
        lag_times_ps.append(visc_data.get("lag_time_ps", []))
        sacf_data.append(visc_data.get("sacf", []))
        viscosity_integral.append(visc_data.get("viscosity_integral", []))

    return {
        "temperatures": sorted_temps,
        "viscosities": viscosities,
        "max_lag": all_max_lags,
        "simulation_steps": sim_steps,
        "lag_times_ps": lag_times_ps,
        "sacf_data": sacf_data,
        "viscosity_integral": viscosity_integral,
    }


# ---------------------------------------------------------------------------
# Parallelised sub-steps  (used by submit_viscosity_substeps)
# ---------------------------------------------------------------------------


def _run_cooling_chain(submission, config, base_result: dict) -> dict:
    """Sequential cooling chain from highest to lowest temperature.

    Returns a dict with ``structures`` (mapping temperature → ASE Atoms)
    and the LAMMPS ``potential`` so that production steps can use them.
    """
    from amorphouspy_api.executor import get_lammps_server_kwargs

    structure = base_result["melt_quench"]["final_structure"]
    potential = base_result["structure_generation"]["potential"]
    lammps_kwargs = get_lammps_server_kwargs()

    sorted_temps = sorted(config.temperatures, reverse=True)
    heating_rate = int(submission.simulation.quench_rate * 100)
    cooling_rate = int(submission.simulation.quench_rate)

    structures: dict[float, Any] = {}
    structure_current = structure

    for idx, temp in enumerate(sorted_temps):
        temp_high = 5000.0 if idx == 0 else sorted_temps[idx - 1]
        logger.info("Viscosity cooling: %.1f K → %.1f K", temp_high, temp)
        mq_result = melt_quench_simulation(
            structure=structure_current,
            potential=potential,
            temperature_high=float(temp_high),
            temperature_low=float(temp),
            timestep=1.0,
            heating_rate=float(heating_rate),
            cooling_rate=float(cooling_rate),
            n_print=1000,
            langevin=False,
            server_kwargs=lammps_kwargs,
        )
        structure_current = mq_result["structure"]
        structures[temp] = structure_current
        logger.info("Cooled to %.1f K, %d atoms", temp, len(structure_current))

    return {"structures": structures, "potential": potential}


def _run_viscosity_at_temp(temp: float, cooling_result: dict, config) -> dict:
    """Run viscosity production + Green-Kubo at a single temperature."""
    from amorphouspy_api.executor import get_lammps_server_kwargs

    structure = cooling_result["structures"][temp]
    potential = cooling_result["potential"]
    lammps_kwargs = get_lammps_server_kwargs()

    logger.info("Running viscosity simulation at %.1f K", temp)
    visc_result = viscosity_simulation(
        structure=structure,
        potential=potential,
        temperature_sim=float(temp),
        timestep=float(config.timestep),
        production_steps=int(config.n_timesteps),
        n_print=int(config.n_print),
        langevin=False,
        seed=12345,
        server_kwargs=lammps_kwargs,
    )

    visc_data = get_viscosity(visc_result, timestep=float(config.timestep), max_lag=config.max_lag)
    logger.info("Viscosity at %.1f K: %.3e Pa·s", temp, visc_data["viscosity"])

    return {
        "temp": float(temp),
        "viscosity": float(visc_data["viscosity"]),
        "max_lag": float(visc_data["max_lag"]),
        "simulation_steps": int(config.n_timesteps),
        "lag_times_ps": visc_data.get("lag_time_ps", []),
        "sacf": visc_data.get("sacf", []),
        "viscosity_integral": visc_data.get("viscosity_integral", []),
    }


def _merge_viscosity(**temp_results: dict) -> dict:
    """Assemble per-temperature results into the final viscosity dict."""
    results = sorted(temp_results.values(), key=lambda r: r["temp"], reverse=True)
    return {
        "viscosity": {
            "temperatures": [r["temp"] for r in results],
            "viscosities": [r["viscosity"] for r in results],
            "max_lag": [r["max_lag"] for r in results],
            "simulation_steps": [r["simulation_steps"] for r in results],
            "lag_times_ps": [r["lag_times_ps"] for r in results],
            "sacf_data": [r["sacf"] for r in results],
            "viscosity_integral": [r["viscosity_integral"] for r in results],
        }
    }


def submit_viscosity_substeps(
    executor: BaseExecutor,
    base_future: Future,
    submission: JobSubmission,
    config: ViscosityAnalysis,
    cache_key: str | None,
    lammps_rd: dict[str, Any],
    base_rd: dict[str, Any],
    *,
    is_slurm: bool,
) -> Future:
    """Submit the viscosity workflow as parallelised executor sub-steps.

    1. **Cooling chain** (sequential, single LAMMPS job):
       cool from highest to lowest temperature, producing one structure per T.
    2. **Production runs** (parallel, one LAMMPS job per temperature):
       ``viscosity_simulation`` + ``get_viscosity`` at each temperature.
    3. **Merge** (lightweight): assemble per-T results into the final dict.

    Returns a future that resolves to ``{"viscosity": { ... }}``.
    """
    sorted_temps = sorted(config.temperatures, reverse=True)

    # --- Step 1: sequential cooling chain ---
    cool_rd: dict[str, Any] = dict(lammps_rd)
    if is_slurm:
        cool_rd["job_name"] = "viscosity_cool"
    if cache_key is not None:
        cool_rd["cache_key"] = f"{cache_key}_viscosity_cool"

    cool_future = executor.submit(
        _run_cooling_chain,
        resource_dict=cool_rd,
        submission=submission,
        config=config,
        base_result=base_future,
    )

    # --- Step 2: parallel production runs (fan-out from cooling) ---
    temp_futures: dict[str, Future] = {}
    for temp in sorted_temps:
        rd: dict[str, Any] = dict(lammps_rd)
        if is_slurm:
            rd["job_name"] = f"viscosity_{temp:.0f}K"
        if cache_key is not None:
            rd["cache_key"] = f"{cache_key}_viscosity_{temp:.0f}K"

        temp_futures[f"T{temp:.0f}"] = executor.submit(
            _run_viscosity_at_temp,
            resource_dict=rd,
            temp=temp,
            cooling_result=cool_future,
            config=config,
        )

    # --- Step 3: merge per-temperature results ---
    merge_rd: dict[str, Any] = dict(base_rd)
    if is_slurm:
        merge_rd["job_name"] = "viscosity_merge"
    if cache_key is not None:
        merge_rd["cache_key"] = f"{cache_key}_viscosity"

    return executor.submit(_merge_viscosity, resource_dict=merge_rd, **temp_futures)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _build_viscosity_vs_temperature_plot(temperatures: list[float], viscosities: list[float]) -> dict:
    """Build Plotly figure dict for viscosity vs temperature in dPa\u00b7s and \u00b0C."""
    temps_c = [t - 273.15 for t in temperatures]
    visc_dpas = [v * 10 for v in viscosities]
    return {
        "data": [
            {
                "x": temps_c,
                "y": visc_dpas,
                "mode": "markers+lines",
                "line": {"width": 2},
                "marker": {"size": 8},
                "name": "Viscosity",
            }
        ],
        "layout": {
            "title": {"text": "Viscosity vs Temperature", "font": {"size": 16}},
            "xaxis": {"title": {"text": "Temperature (\u00b0C)", "font": {"size": 14}}},
            "yaxis": {"title": {"text": "Viscosity (dPa\u00b7s)", "font": {"size": 14}}, "type": "log"},
            "hovermode": "closest",
            "height": 500,
            "margin": {"l": 80, "r": 40, "t": 60, "b": 70},
        },
    }


def _build_arrhenius_plot(temperatures: list[float], viscosities: list[float]) -> dict:
    """Build Plotly figure dict for an Arrhenius-style viscosity vs 1000/T plot."""
    inv_t = [1000.0 / t for t in temperatures]
    visc_dpas = [v * 10 for v in viscosities]
    return {
        "data": [
            {
                "x": inv_t,
                "y": visc_dpas,
                "mode": "markers+lines",
                "line": {"width": 2},
                "marker": {"size": 8},
                "name": "Viscosity",
            }
        ],
        "layout": {
            "title": {"text": "Arrhenius Plot", "font": {"size": 16}},
            "xaxis": {"title": {"text": "1000 / T  (1/K)", "font": {"size": 14}}},
            "yaxis": {"title": {"text": "Viscosity (dPa\u00b7s)", "font": {"size": 14}}, "type": "log"},
            "hovermode": "closest",
            "height": 500,
            "margin": {"l": 80, "r": 40, "t": 60, "b": 70},
        },
    }


def _build_sacf_plot(
    lag_times_ps: list[list[float]],
    sacf_data: list[list[float]],
    temperatures: list[float],
) -> dict | None:
    """Build Plotly figure dict for SACF decay curves (one trace per temperature)."""
    traces = [
        {
            "x": lag_times_ps[i],
            "y": sacf_data[i],
            "mode": "lines",
            "name": f"{temperatures[i]} K",
            "line": {"width": 2},
        }
        for i in range(len(temperatures))
        if lag_times_ps[i] and sacf_data[i]
    ]
    if not traces:
        return None
    return {
        "data": traces,
        "layout": {
            "title": "Stress Autocorrelation Function",
            "xaxis": {"title": "Lag Time (ps)", "type": "log"},
            "yaxis": {"title": "Normalized SACF"},
            "hovermode": "closest",
            "showlegend": True,
            "height": 500,
            "margin": {"l": 80, "r": 40, "t": 60, "b": 60},
        },
    }


def _build_running_viscosity_plot(
    lag_times_ps: list[list[float]],
    viscosity_integral: list[list[float]],
    temperatures: list[float],
) -> dict | None:
    """Build Plotly figure dict for running viscosity convergence curves."""
    traces = [
        {
            "x": lag_times_ps[i],
            "y": [v * 10 for v in viscosity_integral[i]],
            "mode": "lines",
            "name": f"{temperatures[i] - 273.15:.0f} \u00b0C",
            "line": {"width": 2},
        }
        for i in range(len(temperatures))
        if lag_times_ps[i] and viscosity_integral[i]
    ]
    if not traces:
        return None
    return {
        "data": traces,
        "layout": {
            "title": {"text": "Viscosity Convergence", "font": {"size": 16}},
            "xaxis": {"title": {"text": "Lag Time (ps)", "font": {"size": 14}}, "type": "log"},
            "yaxis": {"title": {"text": "Viscosity (dPa\u00b7s)", "font": {"size": 14}}, "type": "log"},
            "hovermode": "closest",
            "showlegend": True,
            "height": 500,
            "margin": {"l": 80, "r": 40, "t": 60, "b": 70},
        },
    }


def prepare_viscosity_plots(visc_data: dict[str, Any]) -> dict[str, str]:
    """Build JSON-encoded Plotly plots from viscosity result data.

    Returns:
        Dict with keys ``arrhenius``, ``sacf`` (optional), ``convergence`` (optional).
    """
    import json

    temps = visc_data.get("temperatures", [])
    viscosities = visc_data.get("viscosities", [])
    lag_times_ps = visc_data.get("lag_times_ps", [])
    visc_data.get("sacf_data", [])
    viscosity_integral = visc_data.get("viscosity_integral", [])

    plots: dict[str, str] = {}
    if temps and viscosities:
        plots["visc_vs_t"] = json.dumps(_build_viscosity_vs_temperature_plot(temps, viscosities))
        plots["arrhenius"] = json.dumps(_build_arrhenius_plot(temps, viscosities))
    conv_fig = _build_running_viscosity_plot(lag_times_ps, viscosity_integral, temps)
    if conv_fig:
        plots["convergence"] = json.dumps(conv_fig)
    return plots
