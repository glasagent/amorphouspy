"""Visualization router for pyiron-glass API.

This module provides endpoints for visualizing simulation results,
including structural analysis plots and data summaries.
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pyiron_glass.workflows.structural_analysis import StructureData

from .database import get_task_store

logger = logging.getLogger(__name__)

# Create visualization router
router = APIRouter(prefix="/viz", tags=["visualization"])


def create_results_html(task_id: str, result_data: dict[str, Any], plotly_fig: dict) -> str:
    """Create HTML page displaying simulation results with interactive Plotly plot.

    Args:
        task_id: Task identifier
        result_data: Dictionary containing simulation results
        plotly_fig: Plotly figure as dictionary

    Returns:
        HTML string for the results page
    """
    structural_analysis = result_data.get("structural_analysis", {})

    # Extract key properties for display
    density = structural_analysis.get("density", "N/A")
    network_connectivity = structural_analysis.get("network", {}).get("connectivity", "N/A")
    network_formers = structural_analysis.get("elements", {}).get("formers", [])
    modifiers = structural_analysis.get("elements", {}).get("modifiers", [])

    # Convert Plotly figure to JSON string
    plotly_json = json.dumps(plotly_fig)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Glass Simulation Results - Task {task_id}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                max-width: 1400px;
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
                grid-template-columns: 1fr 2fr;
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
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            #plotly-div {{
                width: 100%;
                height: 1000px;
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
            .download-buttons {{
                margin-top: 15px;
                text-align: center;
            }}
            .download-btn {{
                background: #667eea;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 5px;
                margin: 0 5px;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
            }}
            .download-btn:hover {{
                background: #556cd6;
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

                <div class="download-buttons">
                    <button class="download-btn" onclick="downloadPlot('png')">Download PNG</button>
                    <button class="download-btn" onclick="downloadPlot('svg')">Download SVG</button>
                </div>
            </div>

            <div class="plot-panel">
                <div class="section-title">Interactive Structural Analysis</div>
                <div id="plotly-div"></div>
                <p style="font-size: 0.9em; color: #666; margin-top: 15px;">
                    Interactive plot with zoom, pan, and hover capabilities. Click legend items to show/hide traces.
                    Comprehensive structural analysis including coordination distributions,
                    network connectivity, bond angles, ring statistics, and radial distribution functions.
                </p>
            </div>
        </div>

        <div class="footer">
            <p>Generated by pyiron-glass API | Structural analysis using PMMCS potential</p>
        </div>

        <script>
            // Render the Plotly plot
            const plotlyData = {plotly_json};
            Plotly.newPlot('plotly-div', plotlyData.data, plotlyData.layout, {{
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
                displaylogo: false
            }});

            // Download functionality
            function downloadPlot(format) {{
                const filename = `glass_analysis_{task_id}`;
                if (format === 'png') {{
                    Plotly.downloadImage('plotly-div', {{
                        format: 'png',
                        width: 1400,
                        height: 1200,
                        filename: filename
                    }});
                }} else if (format === 'svg') {{
                    Plotly.downloadImage('plotly-div', {{
                        format: 'svg',
                        width: 1400,
                        height: 1200,
                        filename: filename
                    }});
                }}
            }}
        </script>
    </body>
    </html>
    """


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

        # Create HTML response
        html_content = create_results_html(task_id, result_data, plotly_fig)

        logger.info("Generated interactive visualization for task %s", task_id)
        return HTMLResponse(content=html_content)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.exception("Error generating visualization for task %s", task_id)
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {e!s}") from e
