"""Structural analysis (RDF, coordination, bond angles) on quenched glass."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

    from amorphouspy_api.models import JobSubmission

logger = logging.getLogger(__name__)


def run_structural_analysis(submission: JobSubmission, config: BaseModel, result: dict) -> dict:
    """Structural analysis (RDF, coordination, bond angles) on the quenched glass."""
    from amorphouspy.workflows.structural_analysis import analyze_structure

    data = analyze_structure(atoms=result["melt_quench"]["final_structure"])
    return data.model_dump() if hasattr(data, "model_dump") else data


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _atoms_to_xyz_string(atoms: object) -> str:
    """Convert ASE Atoms object to extended XYZ format string for 3Dmol.js."""
    if atoms is None:
        return ""
    try:
        from io import StringIO

        from ase.io import write

        from amorphouspy_api.models import validate_atoms

        atoms_obj = validate_atoms(atoms)
        xyz_buffer = StringIO()
        write(xyz_buffer, atoms_obj, format="extxyz")
        return xyz_buffer.getvalue()
    except Exception:
        logger.exception("Error converting atoms to extended XYZ")
        return ""


def prepare_structure_context(result_data: dict[str, Any]) -> dict[str, Any]:
    """Extract structural-analysis display data and Plotly JSON from results.

    Returns a dict with keys consumed by the results template:
    ``plotly_json``, ``structure_xyz``, ``density``, ``network_connectivity``,
    ``network_formers``, ``modifiers``.
    """
    from amorphouspy.workflows.structural_analysis import (
        StructureData,
        plot_analysis_results_plotly,
    )

    structural_analysis = result_data.get("structure_characterization", {})
    mq = result_data.get("melt_quench", {})

    # Build Plotly figure
    sa = structural_analysis
    if isinstance(sa, dict):
        sa = StructureData(**sa)
    plotly_fig = plot_analysis_results_plotly(sa).to_dict()

    # 3D structure XYZ
    structure_xyz = ""
    try:
        structure_xyz = _atoms_to_xyz_string(mq["final_structure"])
    except Exception:
        logger.exception("Could not convert structure to XYZ")

    # Scalar properties
    density = structural_analysis.get("density", "N/A")
    if isinstance(density, (int, float)):
        density = f"{density:.3f}"

    network_connectivity = structural_analysis.get("network", {}).get("connectivity", "N/A")
    if isinstance(network_connectivity, (int, float)):
        network_connectivity = f"{network_connectivity:.3f}"

    return {
        "plotly_json": json.dumps(plotly_fig),
        "structure_xyz": structure_xyz,
        "density": density,
        "network_connectivity": network_connectivity,
        "network_formers": structural_analysis.get("elements", {}).get("formers", []),
        "modifiers": structural_analysis.get("elements", {}).get("modifiers", []),
    }
