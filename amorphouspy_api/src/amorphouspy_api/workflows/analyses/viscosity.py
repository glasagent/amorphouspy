"""Viscosity workflow for glass simulation.

Runs Green-Kubo viscosity calculations at multiple temperatures starting
from an already-quenched glass structure.  The structure is sequentially
cooled from the highest to the lowest requested temperature, and at each
step a production MD run is performed followed by post-processing.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from amorphouspy import melt_quench_simulation
from amorphouspy.workflows.viscosity import get_viscosity, viscosity_simulation

if TYPE_CHECKING:
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
        lag_times_ps.append(_downsample_log(visc_data.get("lag_time_ps", [])))
        viscosity_integral.append(_downsample_log(visc_data.get("viscosity_integral", [])))

    return {
        "temperatures": sorted_temps,
        "viscosities": viscosities,
        "max_lag": all_max_lags,
        "simulation_steps": sim_steps,
        "lag_times_ps": lag_times_ps,
        "viscosity_integral": viscosity_integral,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_PLOT_POINTS = 1000


def _downsample_log(arr: list[float], max_points: int = _MAX_PLOT_POINTS) -> list[float]:
    """Downsample *arr* to *max_points* using log-spaced indices.

    The convergence plot uses a logarithmic x-axis, so log-spaced
    sampling preserves visual fidelity while drastically reducing
    the amount of data stored in the database.
    """
    n = len(arr)
    if n <= max_points:
        return arr
    # log-spaced indices from 0 to n-1, always including the last point
    indices = sorted({round(v) for v in _logspace(0, n - 1, max_points)})
    return [arr[i] for i in indices]


def _logspace(start: float, stop: float, num: int) -> list[float]:
    """Return *num* values log-spaced between *start* and *stop* (inclusive)."""
    if num <= 0:
        return []
    if num == 1:
        return [stop]
    # shift by 1 so log(0) is avoided
    log_start = math.log10(start + 1)
    log_stop = math.log10(stop + 1)
    step = (log_stop - log_start) / (num - 1)
    return [10 ** (log_start + i * step) - 1 for i in range(num)]


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
        Dict with keys ``visc_vs_t``, ``arrhenius``, ``convergence`` (optional).
    """
    import json

    temps = visc_data.get("temperatures", [])
    viscosities = visc_data.get("viscosities", [])
    lag_times_ps = visc_data.get("lag_times_ps", [])
    viscosity_integral = visc_data.get("viscosity_integral", [])

    plots: dict[str, str] = {}
    if temps and viscosities:
        plots["visc_vs_t"] = json.dumps(_build_viscosity_vs_temperature_plot(temps, viscosities))
        plots["arrhenius"] = json.dumps(_build_arrhenius_plot(temps, viscosities))
    conv_fig = _build_running_viscosity_plot(lag_times_ps, viscosity_integral, temps)
    if conv_fig:
        plots["convergence"] = json.dumps(conv_fig)
    return plots
