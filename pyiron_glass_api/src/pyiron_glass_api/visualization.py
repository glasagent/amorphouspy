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
router = APIRouter(prefix="/viz", tags=["visualization"])

# Setup Jinja2 templates
template_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))


def atoms_to_xyz_string(atoms) -> str:
    """Convert ASE Atoms object to XYZ format string for 3Dmol.js.

    Args:
        atoms: ASE Atoms object or serialized structure data

    Returns:
        XYZ format string
    """
    if atoms is None:
        return ""

    try:
        from io import StringIO

        from ase.io import write

        from .models import validate_atoms

        # Use our custom validator to handle any input format
        atoms_obj = validate_atoms(atoms)

        # Convert to XYZ format using ASE's built-in writer
        xyz_buffer = StringIO()
        write(xyz_buffer, atoms_obj, format="xyz")
        return xyz_buffer.getvalue()

    except Exception as e:
        logger.exception("Error converting atoms to XYZ: %s", e)
        logger.exception("Atoms type: %s", type(atoms))
        return ""


def prepare_template_context(task_id: str, result_data: dict[str, Any], plotly_fig: dict, structure_xyz: str = "") -> dict[str, Any]:
    """Prepare context data for the results template.

    Args:
        task_id: Task identifier
        result_data: Dictionary containing simulation results
        plotly_fig: Plotly figure as dictionary
        structure_xyz: XYZ format string for 3D visualization

    Returns:
        Dictionary containing all template variables
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


@router.get("/results/{task_id}", response_class=HTMLResponse)
async def visualize_results(task_id: str) -> HTMLResponse:
    """Visualize simulation results for a given task ID with interactive Plotly plots.

    This endpoint returns an HTML page with interactive structural analysis plots and key results.

    Args:
        task_id: The simulation task identifier

    Returns:
        HTML page with interactive results visualization

    Raises:
        HTTPException: If task not found, not completed, or missing structural analysis
    """
    try:
        # Get task data
        task_data = get_task_store().get(task_id)
        if not task_data:
            raise HTTPException(status_code=404, detail="Task not found")

        # Check if task is completed
        if task_data["state"] != "complete":
            raise HTTPException(
                status_code=400, detail=f"Task is not completed yet. Current state: {task_data['state']}"
            )

        # Get result data
        result_data = task_data.get("result")
        if not result_data:
            raise HTTPException(status_code=404, detail="No results found for this task")

        # Check if structural analysis is available
        structural_analysis = result_data.get("structural_analysis")
        if not structural_analysis:
            raise HTTPException(status_code=404, detail="No structural analysis data found")

        # Convert to StructureData if it's a dict
        if isinstance(structural_analysis, dict):
            structural_data = StructureData(**structural_analysis)
        else:
            structural_data = structural_analysis

        # Generate interactive plot
        from pyiron_glass.workflows.structural_analysis import plot_analysis_results_plotly

        plotly_fig = plot_analysis_results_plotly(structural_data).to_dict()

        # Get atomic structure for 3D visualization
        structure_xyz = ""
        if "final_structure" in result_data:
            try:
                atoms = result_data["final_structure"]
                logger.info("Found final_structure data with type: %s", type(atoms))
                if hasattr(atoms, "__len__"):
                    logger.info("Structure data length/size: %s", len(atoms))
                if hasattr(atoms, "__dict__"):
                    logger.info(
                        "Structure data attributes: %s",
                        list(atoms.__dict__.keys()) if hasattr(atoms, "__dict__") else "No __dict__",
                    )

                # Print the actual structure data to see what we're working with
                if isinstance(atoms, str):
                    logger.info("Structure string preview (first 500 chars): %s", atoms[:500])
                else:
                    logger.info("Structure data preview: %s", str(atoms)[:500])

                structure_xyz = atoms_to_xyz_string(atoms)
                if structure_xyz:
                    logger.info("Successfully converted structure to XYZ format (%d chars)", len(structure_xyz))
                else:
                    logger.warning("XYZ conversion resulted in empty string")
            except Exception as e:
                logger.exception("Could not convert structure to XYZ format: %s", e)
        else:
            logger.warning(
                "No 'final_structure' key found in result_data. Available keys: %s", list(result_data.keys())
            )

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
