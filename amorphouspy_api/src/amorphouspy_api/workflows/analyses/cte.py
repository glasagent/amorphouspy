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
        cte_result = run_cte_fluctuations(
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
        cte_result["metadata"] = {
            "temperature": config.temperature,
            "production_steps": config.production_steps,
            "timestep": config.timestep,
        }
        return cte_result

    cte_result = run_cte_temperature_scan(
        structure=structure,
        potential=potential,
        temperatures=config.temperatures,
        pressure=config.pressure,
        timestep=config.timestep,
        equilibration_steps=config.equilibration_steps,
        production_steps=config.production_steps,
        lammps_resource_dict=resource_dict,
    )
    cte_result["metadata"] = {
        "temperatures": config.temperatures,
        "production_steps": config.production_steps,
        "timestep": config.timestep,
    }
    return cte_result


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


def _build_cte_convergence_plot(data: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Build Plotly figure for CTE convergence (fluctuations method).

    Shows cumulative mean +/- uncertainty for the linear CTE (average of
    CTE_x, CTE_y, CTE_z) in ppm/K versus production run index, with final
    value annotated at the right end.
    """
    run_index = data.get("run_index", [])
    if not run_index:
        return None

    # Compute linear CTE as average of the three directional components
    cte_x = data.get("CTE_x", [])
    cte_y = data.get("CTE_y", [])
    cte_z = data.get("CTE_z", [])
    if not (cte_x and cte_y and cte_z):
        return None
    values = [(x + y + z) / 3.0 for x, y, z in zip(cte_x, cte_y, cte_z, strict=False)]

    # Extract metadata for title and x-axis
    temperature = metadata.get("temperature", "N/A") if metadata else "N/A"
    production_steps = metadata.get("production_steps") if metadata else None
    timestep = metadata.get("timestep", 1.0) if metadata else 1.0
    production_ns = production_steps * timestep * 1e-6 if production_steps else None

    # X-axis: cumulative simulation time in ns (or fall back to run index)
    if production_ns is not None:
        x_values = [i * production_ns for i in run_index]
        x_title = "Simulation Time (ns)"
    else:
        x_values = list(run_index)
        x_title = "Production Run"

    PPM = 1e6  # conversion factor 1/K -> ppm/K
    color = "#7b2d8e"

    means, uncertainties = _cumulative_mean_and_uncertainty(values)
    means_ppm = [m * PPM for m in means]
    unc_ppm = [u * PPM for u in uncertainties]
    upper = [m + u for m, u in zip(means_ppm, unc_ppm, strict=False)]
    lower = [m - u for m, u in zip(means_ppm, unc_ppm, strict=False)]

    traces: list[dict[str, Any]] = [
        {
            "x": x_values,
            "y": means_ppm,
            "mode": "lines",
            "name": "CTE<sub>L</sub>",
            "line": {"color": color, "width": 2},
        },
        {
            "x": [*x_values, *reversed(x_values)],
            "y": [*upper, *reversed(lower)],
            "fill": "toself",
            "fillcolor": color,
            "opacity": 0.2,
            "line": {"width": 0},
            "showlegend": False,
            "hoverinfo": "skip",
        },
    ]

    # Annotation at right end with final CTE value
    final_mean = means_ppm[-1]
    final_unc = unc_ppm[-1]
    annotations = [
        {
            "x": x_values[-1],
            "y": final_mean,
            "xanchor": "left",
            "yanchor": "middle",
            "text": f"<b>{final_mean:.1f} \u00b1 {final_unc:.1f} ppm/K</b>",
            "showarrow": False,
            "font": {"size": 13, "color": color},
            "xshift": 10,
        }
    ]

    # Build title with temperature
    title_parts = ["Linear CTE Convergence"]
    if temperature != "N/A":
        title_parts.append(f"T = {temperature} K")
    title = " \u2014 ".join(title_parts)

    return {
        "data": traces,
        "layout": {
            "title": {"text": title, "font": {"size": 16}},
            "xaxis": {"title": {"text": x_title, "font": {"size": 14}}},
            "yaxis": {"title": {"text": "CTE (ppm/K)", "font": {"size": 14}}},
            "hovermode": "closest",
            "showlegend": True,
            "height": 450,
            "margin": {"l": 80, "r": 120, "t": 60, "b": 60},
            "annotations": annotations,
        },
    }


def _build_cte_summary_plot(summary: dict[str, Any]) -> dict[str, Any] | None:
    """Build Plotly bar chart of final linear CTE value with error bar and temperature."""
    # Compute linear CTE as average of x, y, z components
    x_mean = summary.get("CTE_x_mean")
    y_mean = summary.get("CTE_y_mean")
    z_mean = summary.get("CTE_z_mean")
    x_unc = summary.get("CTE_x_uncertainty", 0.0)
    y_unc = summary.get("CTE_y_uncertainty", 0.0)
    z_unc = summary.get("CTE_z_uncertainty", 0.0)
    temperature = summary.get("temperature", "N/A")

    if x_mean is None or y_mean is None or z_mean is None:
        return None

    mean_val = (x_mean + y_mean + z_mean) / 3.0
    # Propagate uncertainty: s_avg = sqrt(s_x^2 + s_y^2 + s_z^2) / 3
    unc_val = (x_unc**2 + y_unc**2 + z_unc**2) ** 0.5 / 3.0

    title = f"Linear CTE Summary (T = {temperature} K)" if temperature != "N/A" else "Linear CTE Summary"

    return {
        "data": [
            {
                "x": ["CTE<sub>L</sub>"],
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
        conv_fig = _build_cte_convergence_plot(data, metadata=cte_data.get("metadata"))
        if conv_fig:
            plots["convergence"] = json.dumps(conv_fig)
    else:
        # Temperature scan method — top-level keys are "01_300K", etc.
        vt_fig = _build_cte_vt_plot(cte_data)
        if vt_fig:
            plots["volume_temperature"] = json.dumps(vt_fig)

    return plots
