"""Visualization helpers for amorphouspy API.

Renders an interactive HTML page with Plotly charts and 3Dmol.js
for a completed simulation job.
"""

import json
import logging
from pathlib import Path
from typing import Any

from amorphouspy.workflows.structural_analysis import StructureData
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from amorphouspy_api.database import get_job_store
from amorphouspy_api.models import validate_atoms

logger = logging.getLogger(__name__)

template_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))


def atoms_to_xyz_string(atoms) -> str:
    """Convert ASE Atoms object to extended XYZ format string for 3Dmol.js."""
    if atoms is None:
        return ""
    try:
        from io import StringIO

        from ase.io import write

        atoms_obj = validate_atoms(atoms)
        xyz_buffer = StringIO()
        write(xyz_buffer, atoms_obj, format="extxyz")
        return xyz_buffer.getvalue()
    except Exception:
        logger.exception("Error converting atoms to extended XYZ")
        return ""


def prepare_template_context(
    job_id: str, result_data: dict[str, Any], plotly_fig: dict, structure_xyz: str = ""
) -> dict[str, Any]:
    """Prepare context data for the results template."""
    structural_analysis = result_data.get("structure", {})
    mq = result_data.get("melt_quench", {})

    # Extract key properties for display
    density = structural_analysis.get("density", "N/A")
    if isinstance(density, (int, float)):
        density = f"{density:.3f}"

    network_connectivity = structural_analysis.get("network", {}).get("connectivity", "N/A")
    if isinstance(network_connectivity, (int, float)):
        network_connectivity = f"{network_connectivity:.3f}"

    network_formers = structural_analysis.get("elements", {}).get("formers", [])
    modifiers = structural_analysis.get("elements", {}).get("modifiers", [])

    mean_temperature = mq.get("mean_temperature", "N/A")
    if isinstance(mean_temperature, (int, float)):
        mean_temperature = f"{mean_temperature:.1f}"

    simulation_steps = mq.get("simulation_steps", "N/A")
    if isinstance(simulation_steps, int):
        simulation_steps = f"{simulation_steps:,}"

    return {
        "job_id": job_id,
        "composition": mq.get("composition", "N/A"),
        "mean_temperature": mean_temperature,
        "simulation_steps": simulation_steps,
        "density": density,
        "network_connectivity": network_connectivity,
        "network_formers": network_formers,
        "modifiers": modifiers,
        "plotly_json": json.dumps(plotly_fig),
        "structure_xyz": structure_xyz,
    }


def _process_structure_for_3d(result_data: dict) -> str:
    """Process atomic structure data for 3D visualization.

    Args:
        result_data: Result data that may contain final_structure.

    Returns:
        XYZ format string for 3D visualization.
    """
    structure_xyz = ""

    try:
        mq = result_data.get("melt_quench", {})
        atoms = mq["final_structure"]

        structure_xyz = atoms_to_xyz_string(atoms)
        if structure_xyz:
            logger.info("Successfully converted structure to XYZ format (%d chars)", len(structure_xyz))
        else:
            logger.warning("XYZ conversion resulted in empty string")

    except Exception:
        logger.exception("Could not convert structure to XYZ format")

    return structure_xyz


def render_job_visualization(job_id: str) -> HTMLResponse:
    """Render interactive visualisation for a completed job.

    Called from ``routers/jobs.py`` (``GET /jobs/{id}/visualize``).
    """
    store = get_job_store()
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed yet. Current status: {job.status}")

    result_data = job.result_data
    if not result_data:
        raise HTTPException(status_code=404, detail="No results found for this job")

    structural_analysis = result_data.get("structure")
    if not structural_analysis:
        raise HTTPException(status_code=404, detail="No structural analysis data found")

    try:
        from amorphouspy.workflows.structural_analysis import plot_analysis_results_plotly

        if isinstance(structural_analysis, dict):
            structural_analysis = StructureData(**structural_analysis)

        plotly_fig = plot_analysis_results_plotly(structural_analysis).to_dict()
        structure_xyz = _process_structure_for_3d(result_data)

        context = prepare_template_context(job_id, result_data, plotly_fig, structure_xyz)

        # Include viscosity plots if the job ran viscosity analysis
        visc_data = result_data.get("viscosity")
        if visc_data:
            context["viscosity_plots"] = prepare_viscosity_plots(visc_data)

        html_content = templates.get_template("results.html").render(context)

        logger.info("Generated visualisation for job %s", job_id)
        return HTMLResponse(content=html_content)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating visualisation for job %s", job_id)
        raise HTTPException(status_code=500, detail=f"Error generating visualisation: {e!s}") from e


# ---------------------------------------------------------------------------
# Viscosity visualization helpers
# ---------------------------------------------------------------------------


def _build_arrhenius_plot(temperatures: list[float], viscosities: list[float]) -> dict:
    """Build Plotly figure dict for an Arrhenius-style viscosity vs 1000/T plot."""
    inv_t = [1000.0 / t for t in temperatures]
    return {
        "data": [
            {
                "x": inv_t,
                "y": viscosities,
                "mode": "markers+lines",
                "line": {"width": 2},
                "name": "Viscosity",
            }
        ],
        "layout": {
            "title": "Viscosity vs 1000/T",
            "xaxis": {"title": "1000 / T  (1/K)"},
            "yaxis": {"title": "Viscosity (Pa·s)", "type": "log"},
            "hovermode": "closest",
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
            "y": viscosity_integral[i],
            "mode": "lines",
            "name": f"{temperatures[i]} K",
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
            "title": "Viscosity Convergence",
            "xaxis": {"title": "Lag Time (ps)", "type": "log"},
            "yaxis": {"title": "Viscosity (Pa·s)", "type": "log"},
            "hovermode": "closest",
            "showlegend": True,
        },
    }


def prepare_viscosity_plots(visc_data: dict[str, Any]) -> dict[str, str]:
    """Build JSON-encoded Plotly plots from viscosity result data.

    Returns:
        Dict with keys 'arrhenius', 'sacf' (optional), 'convergence' (optional).
    """
    temps = visc_data.get("temperatures", [])
    viscosities = visc_data.get("viscosities", [])
    lag_times_ps = visc_data.get("lag_times_ps", [])
    sacf_data = visc_data.get("sacf_data", [])
    viscosity_integral = visc_data.get("viscosity_integral", [])

    plots: dict[str, str] = {}
    if temps and viscosities:
        plots["arrhenius"] = json.dumps(_build_arrhenius_plot(temps, viscosities))
    sacf_fig = _build_sacf_plot(lag_times_ps, sacf_data, temps)
    if sacf_fig:
        plots["sacf"] = json.dumps(sacf_fig)
    conv_fig = _build_running_viscosity_plot(lag_times_ps, viscosity_integral, temps)
    if conv_fig:
        plots["convergence"] = json.dumps(conv_fig)
    return plots
