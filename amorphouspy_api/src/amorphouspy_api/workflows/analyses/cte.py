"""CTE workflow wrappers for the amorphouspy API.

Thin wrappers around ``amorphouspy.workflows.cte`` that adapt the core
simulation functions to the API pipeline calling convention.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

    from amorphouspy_api.models import JobSubmission

logger = logging.getLogger(__name__)


def run_cte(submission: JobSubmission, config: BaseModel, result: dict) -> dict:
    """CTE analysis via fluctuations or temperature scan."""
    from amorphouspy_api.executor import get_lammps_server_kwargs
    from amorphouspy_api.models import CTEFluctuations

    potential = result["structure_generation"]["potential"]
    structure = result["melt_quench"]["final_structure"]
    resource_dict = get_lammps_server_kwargs()

    if isinstance(config, CTEFluctuations):
        return run_cte_fluctuations(
            structure=structure,
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

    return run_cte_temperature_scan(
        structure=structure,
        potential=potential,
        temperatures=config.temperatures,
        pressure=config.pressure,
        timestep=config.timestep,
        equilibration_steps=config.equilibration_steps,
        production_steps=config.production_steps,
        lammps_resource_dict=resource_dict,
    )


def run_cte_fluctuations(
    structure,
    potential,
    *,
    temperature: float = 300.0,
    pressure: float = 1e-4,
    timestep: float = 1.0,
    equilibration_steps: int = 100_000,
    production_steps: int = 200_000,
    min_production_runs: int = 2,
    max_production_runs: int = 25,
    cte_uncertainty_criterion: float = 1e-6,
    lammps_resource_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the enthalpy-volume fluctuations CTE workflow."""
    from amorphouspy import cte_from_fluctuations_simulation

    logger.info(
        "Running CTE fluctuations at %.1f K (max %d runs, criterion %.1e)",
        temperature,
        max_production_runs,
        cte_uncertainty_criterion,
    )

    return cte_from_fluctuations_simulation(
        structure=structure,
        potential=potential,
        temperature=temperature,
        pressure=pressure,
        timestep=timestep,
        equilibration_steps=equilibration_steps,
        production_steps=production_steps,
        min_production_runs=min_production_runs,
        max_production_runs=max_production_runs,
        CTE_uncertainty_criterion=cte_uncertainty_criterion,
        server_kwargs=lammps_resource_dict or {},
    )


def run_cte_temperature_scan(
    structure,
    potential,
    *,
    temperatures: list[float],
    pressure: float = 1e-4,
    timestep: float = 1.0,
    equilibration_steps: int = 100_000,
    production_steps: int = 200_000,
    lammps_resource_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the temperature-scan CTE workflow."""
    from amorphouspy import temperature_scan_simulation

    logger.info("Running CTE temperature scan at %s K", temperatures)

    return temperature_scan_simulation(
        structure=structure,
        potential=potential,
        temperature=temperatures,
        pressure=pressure,
        timestep=timestep,
        equilibration_steps=equilibration_steps,
        production_steps=production_steps,
        server_kwargs=lammps_resource_dict or {},
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _cumulative_mean_and_uncertainty(
    values: list[float],
) -> tuple[list[float], list[float]]:
    """Compute running mean and standard-error-of-the-mean for a list of values."""
    import math

    means: list[float] = []
    uncertainties: list[float] = []
    running_sum = 0.0
    running_sq_sum = 0.0
    for i, v in enumerate(values, 1):
        running_sum += v
        running_sq_sum += v * v
        mean = running_sum / i
        means.append(mean)
        if i < 2:
            uncertainties.append(0.0)
        else:
            variance = (running_sq_sum / i - mean * mean) * i / (i - 1)
            uncertainties.append(math.sqrt(max(variance, 0.0) / i))
    return means, uncertainties


def _build_cte_convergence_plot(data: dict[str, Any]) -> dict[str, Any] | None:
    """Build Plotly figure for CTE convergence (fluctuations method).

    Shows cumulative mean +/- uncertainty for CTE_V, CTE_x, CTE_y, CTE_z
    versus production run index.
    """
    run_index = data.get("run_index", [])
    if not run_index:
        return None

    traces: list[dict[str, Any]] = []
    colors = {
        "CTE_V": "#7b2d8e",
    }

    for key, color in colors.items():
        values = data.get(key, [])
        if not values:
            continue
        means, uncertainties = _cumulative_mean_and_uncertainty(values)
        upper = [m + u for m, u in zip(means, uncertainties, strict=False)]
        lower = [m - u for m, u in zip(means, uncertainties, strict=False)]

        traces.append(
            {
                "x": run_index,
                "y": means,
                "mode": "lines",
                "name": f"{key}: {means[-1]:.2e} +/- {uncertainties[-1]:.2e} 1/K",
                "line": {"color": color, "width": 2},
            }
        )
        # Uncertainty band (upper + reversed lower)
        traces.append(
            {
                "x": [*run_index, *reversed(run_index)],
                "y": [*upper, *reversed(lower)],
                "fill": "toself",
                "fillcolor": color,
                "opacity": 0.2,
                "line": {"width": 0},
                "showlegend": False,
                "hoverinfo": "skip",
            }
        )

    if not traces:
        return None

    return {
        "data": traces,
        "layout": {
            "title": {"text": "CTE Convergence (Fluctuations)", "font": {"size": 16}},
            "xaxis": {"title": {"text": "Production Run", "font": {"size": 14}}},
            "yaxis": {"title": {"text": "CTE (1/K)", "font": {"size": 14}}, "exponentformat": "e"},
            "hovermode": "closest",
            "showlegend": True,
            "height": 450,
            "margin": {"l": 80, "r": 40, "t": 60, "b": 60},
        },
    }


def _build_cte_summary_plot(summary: dict[str, Any]) -> dict[str, Any] | None:
    """Build Plotly bar chart of final CTE_V value with error bar and temperature."""
    mean_val = summary.get("CTE_V_mean")
    unc_val = summary.get("CTE_V_uncertainty", 0.0)
    temperature = summary.get("temperature", "N/A")

    if mean_val is None:
        return None

    title = f"CTE Summary (T = {temperature} K)" if temperature != "N/A" else "CTE Summary"

    return {
        "data": [
            {
                "x": ["CTE_V"],
                "y": [mean_val],
                "type": "bar",
                "error_y": {"type": "data", "array": [unc_val], "visible": True},
                "marker": {"color": ["#7b2d8e"]},
                "text": [f"{mean_val:.2e} \u00b1 {unc_val:.2e} 1/K"],
                "textposition": "outside",
            }
        ],
        "layout": {
            "title": {"text": title, "font": {"size": 16}},
            "xaxis": {"title": ""},
            "yaxis": {"title": {"text": "CTE (1/K)", "font": {"size": 14}}, "exponentformat": "e"},
            "hovermode": "closest",
            "height": 450,
            "margin": {"l": 80, "r": 40, "t": 60, "b": 40},
        },
    }


def _build_cte_vt_plot(cte_data: dict[str, Any]) -> dict[str, Any] | None:
    """Build Plotly V-T scatter plot for the temperature-scan method.

    Looks for per-temperature keys like ``01_300K`` in the result dict.
    """
    temperatures: list[float] = []
    volumes: list[float] = []

    for key in sorted(cte_data.keys()):
        # Keys look like "01_300K"
        if "_" not in key or "K" not in key:
            continue
        runs = cte_data[key]
        if not isinstance(runs, dict):
            continue
        # Average V across runs at this temperature
        v_vals = [r.get("V") for r in runs.values() if isinstance(r, dict) and r.get("V") is not None]
        if not v_vals:
            continue
        # Parse temperature from key
        try:
            temp = float(key.split("_")[1].rstrip("K"))
        except (IndexError, ValueError):
            continue
        temperatures.append(temp)
        volumes.append(sum(v_vals) / len(v_vals))

    if len(temperatures) < 2:
        return None

    return {
        "data": [
            {
                "x": temperatures,
                "y": volumes,
                "mode": "markers+lines",
                "line": {"width": 2},
                "name": "V vs T",
            }
        ],
        "layout": {
            "title": "Volume vs Temperature",
            "xaxis": {"title": "Temperature (K)"},
            "yaxis": {"title": "Volume (\u00c5\u00b3)"},
            "hovermode": "closest",
        },
    }


def prepare_cte_plots(cte_data: dict[str, Any]) -> dict[str, str]:
    """Build JSON-encoded Plotly plots from CTE result data.

    Returns:
        Dict with keys ``convergence`` and/or ``summary`` (fluctuations),
        or ``volume_temperature`` (temperature scan).
    """
    import json

    plots: dict[str, str] = {}

    # Fluctuations method produces "summary" + "data" keys
    summary = cte_data.get("summary")
    data = cte_data.get("data")

    if summary and data:
        conv_fig = _build_cte_convergence_plot(data)
        if conv_fig:
            plots["convergence"] = json.dumps(conv_fig)
        summ_fig = _build_cte_summary_plot(summary)
        if summ_fig:
            plots["summary"] = json.dumps(summ_fig)
    else:
        # Temperature scan method — top-level keys are "01_300K", etc.
        vt_fig = _build_cte_vt_plot(cte_data)
        if vt_fig:
            plots["volume_temperature"] = json.dumps(vt_fig)

    return plots
