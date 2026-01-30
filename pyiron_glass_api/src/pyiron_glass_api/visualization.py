"""Visualization router for pyiron-glass API.

This module provides endpoints for visualizing simulation results,
including structural analysis plots and data summaries.
"""

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pyiron_glass.workflows.structural_analysis import StructureData

from .database import get_task_store

logger = logging.getLogger(__name__)

# Create visualization router
router = APIRouter(prefix="/visualize", tags=["visualization"])

# Setup Jinja2 templates
template_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))


def atoms_to_xyz_string(atoms) -> str:
    """Convert ASE Atoms object to extended XYZ format string for 3Dmol.js.

    Args:
        atoms: ASE Atoms object or serialized structure data.

    Returns:
        Extended XYZ format string with cell information.
    """
    if atoms is None:
        return ""

    try:
        from io import StringIO

        from ase.io import write

        from .models import validate_atoms

        # Use our custom validator to handle any input format
        atoms_obj = validate_atoms(atoms)

        # Use extended XYZ format which naturally includes cell information
        xyz_buffer = StringIO()
        write(xyz_buffer, atoms_obj, format="extxyz")
        return xyz_buffer.getvalue()

    except Exception:
        logger.exception("Error converting atoms to extended XYZ")
        logger.exception("Atoms type: %s", type(atoms))
        return ""


def prepare_template_context(
    task_id: str, result_data: dict[str, Any], plotly_fig: dict, structure_xyz: str = ""
) -> dict[str, Any]:
    """Prepare context data for the results template.

    Args:
        task_id: Task identifier.
        result_data: Dictionary containing simulation results.
        plotly_fig: Plotly figure as dictionary.
        structure_xyz: XYZ format string for 3D visualization.

    Returns:
        Dictionary containing all template variables.
    """
    structural_analysis = result_data.get("structural_analysis", {})

    # Extract key properties for display
    density = structural_analysis.get("density", "N/A")
    if isinstance(density, (int, float)):
        density = f"{density:.3f}"

    network_connectivity = structural_analysis.get("network", {}).get("connectivity", "N/A")
    if isinstance(network_connectivity, (int, float)):
        network_connectivity = f"{network_connectivity:.3f}"

    network_formers = structural_analysis.get("elements", {}).get("formers", [])
    modifiers = structural_analysis.get("elements", {}).get("modifiers", [])

    mean_temperature = result_data.get("mean_temperature", "N/A")
    if isinstance(mean_temperature, (int, float)):
        mean_temperature = f"{mean_temperature:.1f}"

    simulation_steps = result_data.get("simulation_steps", "N/A")
    if isinstance(simulation_steps, int):
        simulation_steps = f"{simulation_steps:,}"

    return {
        "task_id": task_id,
        "composition": result_data.get("composition", "N/A"),
        "mean_temperature": mean_temperature,
        "simulation_steps": simulation_steps,
        "density": density,
        "network_connectivity": network_connectivity,
        "network_formers": network_formers,
        "modifiers": modifiers,
        "plotly_json": json.dumps(plotly_fig),
        "structure_xyz": structure_xyz,
    }


def _validate_task_data(task_data: dict | None, task_id: str) -> dict:
    """Validate task data and ensure it's complete.

    Args:
        task_data: Task data from the store.
        task_id: Task identifier for error messages.

    Returns:
        Validated task data.

    Raises:
        HTTPException: If task not found or not completed.
    """
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")

    if task_data["state"] != "complete":
        raise HTTPException(status_code=400, detail=f"Task is not completed yet. Current state: {task_data['state']}")

    return task_data


def _get_and_validate_results(task_data: dict) -> dict:
    """Extract and validate result data from task.

    Args:
        task_data: Validated task data.

    Returns:
        Result data dictionary.

    Raises:
        HTTPException: If no results or structural analysis found.
    """
    result_data = task_data.get("result")
    if not result_data:
        raise HTTPException(status_code=404, detail="No results found for this task")

    structural_analysis = result_data.get("structural_analysis")
    if not structural_analysis:
        raise HTTPException(status_code=404, detail="No structural analysis data found")

    return result_data


def _process_structure_for_3d(result_data: dict) -> str:
    """Process atomic structure data for 3D visualization.

    Args:
        result_data: Result data that may contain final_structure.

    Returns:
        XYZ format string for 3D visualization.
    """
    structure_xyz = ""

    try:
        atoms = result_data["final_structure"]
        logger.info("Found final_structure data with type: %s", type(atoms))
        logger.info("Structure data preview: %s", str(atoms)[:500])

        structure_xyz = atoms_to_xyz_string(atoms)
        if structure_xyz:
            logger.info("Successfully converted structure to XYZ format (%d chars)", len(structure_xyz))
        else:
            logger.warning("XYZ conversion resulted in empty string")

    except Exception:
        logger.exception("Could not convert structure to XYZ format")

    return structure_xyz


@router.get("/meltquench/{task_id}", response_class=HTMLResponse)
async def visualize_results(task_id: str) -> HTMLResponse:
    """Visualize simulation results for a given task ID with interactive Plotly plots.

    This endpoint returns an HTML page with interactive structural analysis plots and key results.

    Args:
        task_id: The simulation task identifier.

    Returns:
        HTML page with interactive results visualization.

    Raises:
        HTTPException: If task not found, not completed, or missing structural analysis.
    """
    try:
        # Get task data
        task_data = get_task_store().get(task_id)
        task_data = _validate_task_data(task_data, task_id)

        # Get result data
        result_data = _get_and_validate_results(task_data)

        # Generate interactive plot
        from pyiron_glass.workflows.structural_analysis import plot_analysis_results_plotly

        structural_data = result_data["structural_analysis"]
        if isinstance(structural_data, dict):
            structural_data = StructureData(**structural_data)

        plotly_fig = plot_analysis_results_plotly(structural_data).to_dict()

        # Get atomic structure for 3D visualization
        structure_xyz = _process_structure_for_3d(result_data)

        # Create HTML response using template
        context = prepare_template_context(task_id, result_data, plotly_fig, structure_xyz)
        html_content = templates.get_template("results.html").render(context)

        logger.info("Generated interactive visualization for task %s", task_id)
        return HTMLResponse(content=html_content)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.exception("Error generating visualization for task %s", task_id)
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {e!s}") from e


@router.get("/viscosity/{task_id}", response_class=HTMLResponse)
async def visualize_viscosity(task_id: str) -> HTMLResponse:
    """Visualize viscosity results for a given task ID as viscosity vs 1000/T."""

    def validate_and_extract_data(task_data: dict) -> dict:
        task_data = _validate_task_data(task_data, task_id)
        result_data = task_data.get("result")
        if not result_data or result_data.get("kind") != "viscosity":
            raise HTTPException(status_code=404, detail="Task does not contain viscosity results")
        temperatures = result_data.get("temperatures", [])
        viscosities = result_data.get("viscosities", [])
        lag_times_ps = result_data.get("lag_times_ps", [])
        sacf_data = result_data.get("sacf_data", [])
        viscosity_running = result_data.get("viscosity_running", [])
        if not temperatures or not viscosities or len(temperatures) != len(viscosities):
            raise HTTPException(status_code=404, detail="Invalid viscosity result data")
        return {
            "temperatures": temperatures,
            "viscosities": viscosities,
            "lag_times_ps": lag_times_ps,
            "sacf_data": sacf_data,
            "viscosity_running": viscosity_running,
        }

    def build_plot(
        x: list[float],
        y: list[float],
        title: str,
        xaxis_title: str,
        yaxis_title: str,
        *,
        mode: str = "lines",
        log_y: bool = False,
    ) -> dict:
        return {
            "data": [{"x": x, "y": y, "mode": mode, "line": {"width": 2}}],
            "layout": {
                "title": title,
                "xaxis": {"title": xaxis_title, "type": "log" if "Lag" in xaxis_title else "linear"},
                "yaxis": {"title": yaxis_title, "type": "log" if log_y else "linear"},
                "hovermode": "closest",
                "showlegend": True,
            },
        }

    def build_multi_trace_plot(
        x_list: list[list[float]],
        y_list: list[list[float]],
        labels: list[float],
        title: str,
        xaxis_title: str,
        yaxis_title: str,
        *,
        log_y: bool = False,
    ) -> list[dict]:
        traces = [
            {"x": x_list[i], "y": y_list[i], "mode": "lines", "name": f"{labels[i]} K", "line": {"width": 2}}
            for i in range(len(labels))
            if x_list[i] and y_list[i]
        ]
        if traces:
            return [
                {
                    "data": traces,
                    "layout": {
                        "title": title,
                        "xaxis": {"title": xaxis_title, "type": "log" if "Lag" in xaxis_title else "linear"},
                        "yaxis": {"title": yaxis_title, "type": "log" if log_y else "linear"},
                        "hovermode": "closest",
                        "showlegend": True,
                    },
                }
            ]
        return []

    try:
        task_data = get_task_store().get(task_id)
        data = validate_and_extract_data(task_data)

        inv_T = [1000.0 / float(T) for T in data["temperatures"]]
        plotly_json_visc = json.dumps(
            build_plot(
                inv_T,
                data["viscosities"],
                "Viscosity vs 1000/T",
                "1000 / T (1/K)",
                "Viscosity (Pa·s)",
                mode="markers+lines",
                log_y=True,
            )
        )

        sacf_plots = build_multi_trace_plot(
            data["lag_times_ps"],
            data["sacf_data"],
            data["temperatures"],
            "Stress Autocorrelation Function",
            "Lag Time (ps)",
            "Normalized SACF",
        )
        viscosity_running_plots = build_multi_trace_plot(
            data["lag_times_ps"],
            data["viscosity_running"],
            data["temperatures"],
            "Viscosity Convergence",
            "Lag Time (ps)",
            "Viscosity (Pa·s)",
            log_y=True,
        )

        context = {
            "task_id": task_id,
            "temperatures": data["temperatures"],
            "viscosities": data["viscosities"],
            "plotly_json_visc": plotly_json_visc,
            "sacf_plots": [json.dumps(plot) for plot in sacf_plots],
            "viscosity_running_plots": [json.dumps(plot) for plot in viscosity_running_plots],
        }

        html_content = templates.get_template("viscosity_results.html").render(context)
        logger.info("Generated viscosity visualization for task %s", task_id)
        return HTMLResponse(content=html_content)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating viscosity visualization for task %s", task_id)
        raise HTTPException(status_code=500, detail=f"Error generating viscosity visualization: {e!s}") from e
