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

from .database import get_job_store
from .models import validate_atoms

logger = logging.getLogger(__name__)

template_dir = Path(__file__).parent / "templates"
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
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")
    if task_data.get("state", task_data.get("status")) not in ("complete", "completed"):
        state = task_data.get("state", task_data.get("status", "unknown"))
        raise HTTPException(status_code=400, detail=f"Task is not completed yet. Current state: {state}")
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

    structural_analysis = result_data.get("structural_analysis")
    if not structural_analysis:
        raise HTTPException(status_code=404, detail="No structural analysis data found")

    try:
        from amorphouspy.workflows.structural_analysis import plot_analysis_results_plotly

        if isinstance(structural_analysis, dict):
            structural_analysis = StructureData(**structural_analysis)

        plotly_fig = plot_analysis_results_plotly(structural_analysis).to_dict()
        structure_xyz = _process_structure_for_3d(result_data)

        context = prepare_template_context(job_id, result_data, plotly_fig, structure_xyz)
        html_content = templates.get_template("results.html").render(context)

        logger.info("Generated visualisation for job %s", job_id)
        return HTMLResponse(content=html_content)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating visualisation for job %s", job_id)
        raise HTTPException(status_code=500, detail=f"Error generating visualisation: {e!s}") from e
