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

from amorphouspy.pipelines.viscosity import run_viscosity_workflow

if TYPE_CHECKING:
    from amorphouspy_api.models import JobSubmission, ViscosityAnalysis

logger = logging.getLogger(__name__)


def run_viscosity(submission: JobSubmission, config: ViscosityAnalysis, result: dict) -> dict:
    """Multi-temperature viscosity analysis on the quenched glass."""
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


# ---------------------------------------------------------------------------
# VFT fit: log10(η) = A + B / (T - T0)
# ---------------------------------------------------------------------------


def _fit_vft(temperatures_k: list[float], viscosities_dpas: list[float]) -> dict[str, float] | None:
    """Fit Vogel-Fulcher-Tammann (VFT) equation to viscosity data.

    Returns dict with keys ``A``, ``B``, ``T0`` or *None* if the fit fails
    (e.g. fewer than 3 data points or non-convergence).
    """
    import numpy as np
    from scipy.optimize import curve_fit

    if len(temperatures_k) < 3 or len(viscosities_dpas) < 3:
        return None

    t_arr = np.asarray(temperatures_k, dtype=float)
    log_eta = np.log10(np.asarray(viscosities_dpas, dtype=float))

    def vft(t: np.ndarray, a: float, b: float, t0: float) -> np.ndarray:
        return a + b / (t - t0)

    # Initial guess: T0 below the minimum T, A ~ min(log_eta), B from slope
    t0_guess = min(t_arr) - 300.0
    a_guess = float(log_eta.min()) - 1.0
    b_guess = float((log_eta.max() - log_eta.min()) * (min(t_arr) - t0_guess))

    try:
        popt, _ = curve_fit(
            vft,
            t_arr,
            log_eta,
            p0=[a_guess, b_guess, t0_guess],
            bounds=([-np.inf, 0, 0], [np.inf, np.inf, min(t_arr) - 1]),
            maxfev=10_000,
        )
        return {"A": float(popt[0]), "B": float(popt[1]), "T0": float(popt[2])}
    except Exception:
        return None


def _vft_curve(
    vft: dict[str, float],
    t_min_k: float,
    t_max_k: float,
    n_points: int = 200,
) -> tuple[list[float], list[float]]:
    """Evaluate VFT over a temperature range.  Returns (temps_K, visc_dpas)."""
    import numpy as np

    temps = np.linspace(t_min_k, t_max_k, n_points)
    # Guard against T near T0
    mask = temps > vft["T0"] + 1.0
    temps = temps[mask]
    log_eta = vft["A"] + vft["B"] / (temps - vft["T0"])
    return temps.tolist(), (10.0**log_eta).tolist()


def _t_at_log_viscosity(vft: dict[str, float], log_target: float) -> float | None:
    """Solve VFT for the temperature (K) at which log10(η) = *log_target*.

    Returns None if the result is below T0 or non-physical.
    """
    denom = log_target - vft["A"]
    if denom <= 0:
        return None
    t = vft["T0"] + vft["B"] / denom
    if t <= vft["T0"]:
        return None
    return t


# Reference viscosity points (log10 in dPa·s)
_REFERENCE_POINTS = {"T4": 4.0, "T7.6": 7.6}


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _build_viscosity_vs_temperature_plot(
    temperatures: list[float],
    viscosities: list[float],
    vft: dict[str, float] | None = None,
) -> dict:
    """Build Plotly figure dict for viscosity vs temperature in dPa·s and °C."""
    temps_c = [t - 273.15 for t in temperatures]
    visc_dpas = [v * 10 for v in viscosities]

    log_min = math.log10(min(visc_dpas))
    log_max = math.log10(max(visc_dpas))
    log_range = log_max - log_min
    y_upper = log_max + 2 * log_range  # triple the log range towards higher viscosities

    traces: list[dict] = [
        {
            "x": temps_c,
            "y": visc_dpas,
            "mode": "markers+lines",
            "line": {"width": 2},
            "marker": {"size": 8},
            "name": "Green-Kubo",
        }
    ]

    annotations: list[dict] = []
    shapes: list[dict] = []

    if vft is not None:
        # Extrapolation curve down to T where log_eta = y_upper (or T0+10)
        t_at_upper = _t_at_log_viscosity(vft, y_upper)
        t_extrap_min = max(vft["T0"] + 10, t_at_upper or vft["T0"] + 10)
        t_extrap_max = max(temperatures) + 100
        curve_t, curve_eta = _vft_curve(vft, t_extrap_min, t_extrap_max)
        if curve_t:
            traces.append(
                {
                    "x": [t - 273.15 for t in curve_t],
                    "y": curve_eta,
                    "mode": "lines",
                    "line": {"width": 2, "dash": "dash", "color": "rgba(100,100,100,0.6)"},
                    "name": "VFT extrapolation",
                }
            )

        # Reference viscosity point annotations
        for label, log_v in _REFERENCE_POINTS.items():
            if log_v > y_upper:
                continue
            t_ref = _t_at_log_viscosity(vft, log_v)
            if t_ref is None:
                continue
            t_ref_c = t_ref - 273.15
            x_lo = min([*temps_c, t_ref_c]) - 50
            shapes.append(
                {
                    "type": "line",
                    "x0": x_lo,
                    "x1": t_ref_c,
                    "y0": 10**log_v,
                    "y1": 10**log_v,
                    "line": {"color": "rgba(150,150,150,0.5)", "width": 1, "dash": "dot"},
                }
            )
            annotations.append(
                {
                    "x": t_ref_c,
                    "y": log_v,
                    "text": f"{label} = {t_ref_c:.0f} °C",
                    "showarrow": True,
                    "arrowhead": 2,
                    "ax": 40,
                    "ay": -25,
                    "font": {"size": 11},
                }
            )

    layout: dict = {
        "title": {"text": "Viscosity vs Temperature", "font": {"size": 16}},
        "xaxis": {"title": {"text": "Temperature (°C)", "font": {"size": 14}}},
        "yaxis": {
            "title": {"text": "Viscosity (dPa·s)", "font": {"size": 14}},
            "type": "log",
            "exponentformat": "e",
            "range": [log_min - 0.2, y_upper],
        },
        "hovermode": "closest",
        "height": 500,
        "margin": {"l": 80, "r": 40, "t": 60, "b": 70},
    }
    if annotations:
        layout["annotations"] = annotations
    if shapes:
        layout["shapes"] = shapes

    return {"data": traces, "layout": layout}


def _build_arrhenius_plot(
    temperatures: list[float],
    viscosities: list[float],
    vft: dict[str, float] | None = None,
) -> dict:
    """Build Plotly figure dict for an Arrhenius-style viscosity vs 1000/T plot."""
    inv_t = [1000.0 / t for t in temperatures]
    visc_dpas = [v * 10 for v in viscosities]

    log_min = math.log10(min(visc_dpas))
    log_max = math.log10(max(visc_dpas))
    log_range = log_max - log_min
    y_upper = log_max + 2 * log_range  # triple the log range towards higher viscosities

    traces: list[dict] = [
        {
            "x": inv_t,
            "y": visc_dpas,
            "mode": "markers+lines",
            "line": {"width": 2},
            "marker": {"size": 8},
            "name": "Green-Kubo",
        }
    ]

    annotations: list[dict] = []
    shapes: list[dict] = []

    if vft is not None:
        t_at_upper = _t_at_log_viscosity(vft, y_upper)
        t_extrap_min = max(vft["T0"] + 10, t_at_upper or vft["T0"] + 10)
        t_extrap_max = max(temperatures) + 100
        curve_t, curve_eta = _vft_curve(vft, t_extrap_min, t_extrap_max)
        if curve_t:
            traces.append(
                {
                    "x": [1000.0 / t for t in curve_t],
                    "y": curve_eta,
                    "mode": "lines",
                    "line": {"width": 2, "dash": "dash", "color": "rgba(100,100,100,0.6)"},
                    "name": "VFT extrapolation",
                }
            )

        for label, log_v in _REFERENCE_POINTS.items():
            if log_v > y_upper:
                continue
            t_ref = _t_at_log_viscosity(vft, log_v)
            if t_ref is None:
                continue
            inv_t_ref = 1000.0 / t_ref
            x_hi = max([*inv_t, inv_t_ref]) + 0.1
            shapes.append(
                {
                    "type": "line",
                    "x0": inv_t_ref,
                    "x1": x_hi,
                    "y0": 10**log_v,
                    "y1": 10**log_v,
                    "line": {"color": "rgba(150,150,150,0.5)", "width": 1, "dash": "dot"},
                }
            )
            t_ref_c = t_ref - 273.15
            annotations.append(
                {
                    "x": inv_t_ref,
                    "y": log_v,
                    "text": f"{label} = {t_ref_c:.0f} °C",
                    "showarrow": True,
                    "arrowhead": 2,
                    "ax": -40,
                    "ay": -25,
                    "font": {"size": 11},
                }
            )

    layout: dict = {
        "title": {"text": "Arrhenius Plot", "font": {"size": 16}},
        "xaxis": {"title": {"text": "1000 / T  (1/K)", "font": {"size": 14}}},
        "yaxis": {
            "title": {"text": "Viscosity (dPa·s)", "font": {"size": 14}},
            "type": "log",
            "exponentformat": "e",
            "range": [log_min - 0.2, y_upper],
        },
        "hovermode": "closest",
        "height": 500,
        "margin": {"l": 80, "r": 40, "t": 60, "b": 70},
    }
    if annotations:
        layout["annotations"] = annotations
    if shapes:
        layout["shapes"] = shapes

    return {"data": traces, "layout": layout}


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
            "yaxis": {
                "title": {"text": "Viscosity (dPa\u00b7s)", "font": {"size": 14}},
                "type": "log",
                "exponentformat": "e",
            },
            "hovermode": "closest",
            "showlegend": True,
            "height": 500,
            "margin": {"l": 80, "r": 40, "t": 60, "b": 70},
        },
    }


def prepare_viscosity_plots(visc_data: dict[str, Any]) -> dict[str, str]:
    """Build JSON-encoded Plotly plots from viscosity result data.

    Returns:
        Dict with keys ``visc_vs_t``, ``arrhenius``, ``convergence`` (optional),
        and ``vft`` (VFT fit coefficients dict, or None).
    """
    import json

    temps = visc_data.get("temperatures", [])
    viscosities = visc_data.get("viscosities", [])
    lag_times_ps = visc_data.get("lag_times_ps", [])
    viscosity_integral = visc_data.get("viscosity_integral", [])

    plots: dict[str, Any] = {}
    if temps and viscosities:
        visc_dpas = [v * 10 for v in viscosities]
        vft = _fit_vft(temps, visc_dpas)
        plots["visc_vs_t"] = json.dumps(_build_viscosity_vs_temperature_plot(temps, viscosities, vft))
        plots["arrhenius"] = json.dumps(_build_arrhenius_plot(temps, viscosities, vft))
        if vft is not None:
            # Also compute reference temperatures
            ref_temps = {}
            for label, log_v in _REFERENCE_POINTS.items():
                t_ref = _t_at_log_viscosity(vft, log_v)
                if t_ref is not None:
                    ref_temps[label] = round(t_ref - 273.15, 1)
            plots["vft"] = {**{k: round(v, 4) for k, v in vft.items()}, "ref_temps": ref_temps}
    conv_fig = _build_running_viscosity_plot(lag_times_ps, viscosity_integral, temps)
    if conv_fig:
        plots["convergence"] = json.dumps(conv_fig)
    return plots
