"""Visualization router for pyiron-glass API.

This module provides endpoints for visualizing simulation results,
including structural analysis plots and data summaries.
"""

import base64
import io
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pyiron_glass.workflows.structural_analysis import StructureData

from .database import get_task_store

logger = logging.getLogger(__name__)

# Create visualization router
router = APIRouter(prefix="/viz", tags=["visualization"])

# Get task store instance
_task_store = get_task_store()


def generate_plot_from_structural_data(structural_data: StructureData) -> str:
    """Generate a base64-encoded plot from structural analysis data.

    Args:
        structural_data: StructureData object containing structural analysis results

    Returns:
        Base64-encoded PNG image of the plot
    """
    # Import plotting dependencies locally to avoid startup overhead
    import matplotlib as mpl

    mpl.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from pyiron_glass.workflows.structural_analysis import plot_analysis_results

    # Generate the plot
    fig = plot_analysis_results(structural_data)

    # Convert plot to base64 string
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
    img_buffer.seek(0)

    # Encode as base64
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

    # Clean up
    plt.close(fig)
    img_buffer.close()

    return img_base64


def create_results_html(task_id: str, result_data: dict[str, Any], plot_img: str) -> str:
    """Create HTML page displaying simulation results and plot.

    Args:
        task_id: Task identifier
        result_data: Dictionary containing simulation results
        plot_img: Base64-encoded plot image

    Returns:
        HTML string for the results page
    """
    structural_analysis = result_data.get("structural_analysis", {})

    # Extract key properties for display
    density = structural_analysis.get("density", "N/A")
    network_connectivity = structural_analysis.get("network", {}).get("connectivity", "N/A")
    network_formers = structural_analysis.get("elements", {}).get("formers", [])
    modifiers = structural_analysis.get("elements", {}).get("modifiers", [])

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Glass Simulation Results - Task {task_id}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .results-container {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-bottom: 30px;
            }}
            .info-panel {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .plot-panel {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .plot-panel img {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
            }}
            .property {{
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid #eee;
            }}
            .property:last-child {{
                border-bottom: none;
            }}
            .property-label {{
                font-weight: 600;
                color: #555;
            }}
            .property-value {{
                color: #333;
                font-family: 'Courier New', monospace;
            }}
            .section-title {{
                font-size: 1.2em;
                font-weight: bold;
                color: #333;
                margin-bottom: 15px;
                border-bottom: 2px solid #667eea;
                padding-bottom: 5px;
            }}
            .element-list {{
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
            }}
            .element-tag {{
                background: #667eea;
                color: white;
                padding: 4px 8px;
                border-radius: 15px;
                font-size: 0.9em;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding: 20px;
                color: #666;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧪 Glass Simulation Results</h1>
            <p>Task ID: {task_id}</p>
            <p>Composition: {result_data.get("composition", "N/A")}</p>
        </div>

        <div class="results-container">
            <div class="info-panel">
                <div class="section-title">Simulation Results</div>
                <div class="property">
                    <span class="property-label">Composition:</span>
                    <span class="property-value">{result_data.get("composition", "N/A")}</span>
                </div>
                <div class="property">
                    <span class="property-label">Mean Temperature:</span>
                    <span class="property-value">{result_data.get("mean_temperature", "N/A"):.1f} K</span>
                </div>
                <div class="property">
                    <span class="property-label">Simulation Steps:</span>
                    <span class="property-value">{result_data.get("simulation_steps", "N/A"):,}</span>
                </div>

                <div class="section-title" style="margin-top: 25px;">Structural Properties</div>
                <div class="property">
                    <span class="property-label">Density:</span>
                    <span class="property-value">{density:.3f} g/cm³</span>
                </div>
                <div class="property">
                    <span class="property-label">Network Connectivity:</span>
                    <span class="property-value">{network_connectivity:.3f}</span>
                </div>

                <div class="section-title" style="margin-top: 25px;">Element Classification</div>
                <div class="property">
                    <span class="property-label">Network Formers:</span>
                    <div class="element-list">
                        {" ".join(f'<span class="element-tag">{elem}</span>' for elem in network_formers)}
                    </div>
                </div>
                <div class="property">
                    <span class="property-label">Modifiers:</span>
                    <div class="element-list">
                        {" ".join(f'<span class="element-tag">{elem}</span>' for elem in modifiers)}
                    </div>
                </div>
            </div>

            <div class="plot-panel">
                <div class="section-title">Structural Analysis Plot</div>
                <img src="data:image/png;base64,{plot_img}" alt="Structural Analysis Plot">
                <p style="font-size: 0.9em; color: #666; margin-top: 15px;">
                    Comprehensive structural analysis including coordination distributions,
                    network connectivity, bond angles, ring statistics, and radial distribution functions.
                </p>
            </div>
        </div>

        <div class="footer">
            <p>Generated by pyiron-glass API | Structural analysis using PMMCS potential</p>
        </div>
    </body>
    </html>
    """


@router.get("/results/{task_id}", response_class=HTMLResponse)
async def visualize_results(task_id: str) -> HTMLResponse:
    """Visualize simulation results for a given task ID.

    This endpoint returns an HTML page with structural analysis plots and key results.

    Args:
        task_id: The simulation task identifier

    Returns:
        HTML page with results visualization

    Raises:
        HTTPException: If task not found, not completed, or missing structural analysis
    """
    try:
        # Get task data
        task_data = _task_store.get(task_id)
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

        # Generate plot
        plot_img = generate_plot_from_structural_data(structural_data)

        # Create HTML response
        html_content = create_results_html(task_id, result_data, plot_img)

        logger.info("Generated visualization for task %s", task_id)
        return HTMLResponse(content=html_content)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.exception("Error generating visualization for task %s", task_id)
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {e!s}") from e
