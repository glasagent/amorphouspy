"""Structural analysis (RDF, coordination, bond angles) on quenched glass."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ase import Atoms

    from amorphouspy_api.models import JobSubmission, StructureAnalysis

logger = logging.getLogger(__name__)


def run_structural_analysis(submission: JobSubmission, config: StructureAnalysis, result: dict) -> dict:
    """Structural analysis (RDF, coordination, bond angles) on the quenched glass."""
    from amorphouspy.workflows.structural_analysis import analyze_structure

    data = analyze_structure(atoms=result["melt_quench"]["final_structure"])
    return data.model_dump() if hasattr(data, "model_dump") else data


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _atoms_to_xyz_string(atoms: Atoms | dict | str | None) -> str:
    """Convert ASE Atoms object to extended XYZ format string for 3Dmol.js."""
    if atoms is None:
        return ""
    try:
        from io import StringIO

        from ase.io import write

        from amorphouspy_api.models import validate_atoms

        atoms_obj = validate_atoms(atoms)
        if atoms_obj is None:
            return ""
        xyz_buffer = StringIO()
        write(xyz_buffer, atoms_obj, format="extxyz")
        return xyz_buffer.getvalue()
    except Exception:
        logger.exception("Error converting atoms to extended XYZ")
        return ""


def _annotate_atoms_for_viewer(
    atoms_obj: Atoms,
    elements_info: dict[str, Any],
) -> Atoms:
    """Add per-atom ``role`` and ``o_class`` arrays plus ``former_cutoffs`` info.

    Modifies *atoms_obj* in place and returns it.
    """
    import numpy as np

    formers_set = set(elements_info.get("formers", []))
    modifiers_set = set(elements_info.get("modifiers", []))
    oxygen_ids = elements_info.get("oxygen_ids", {})

    # Build reverse map: atom_id -> o_class
    id_to_oclass: dict[int, str] = {}
    for cls, ids in oxygen_ids.items():
        for aid in ids:
            id_to_oclass[aid] = cls

    symbols = atoms_obj.get_chemical_symbols()
    n = len(atoms_obj)
    role = np.empty(n, dtype="U8")
    o_class = np.empty(n, dtype="U4")

    if "id" in atoms_obj.arrays:
        raw_ids = atoms_obj.arrays["id"].astype(np.int64)
    else:
        raw_ids = np.arange(1, n + 1, dtype=np.int64)

    role_map = dict.fromkeys(formers_set, "former")
    role_map.update(dict.fromkeys(modifiers_set, "modifier"))
    role_map["O"] = "oxygen"

    for i, (sym, aid) in enumerate(zip(symbols, raw_ids, strict=False)):
        role[i] = role_map.get(sym, "other")
        o_class[i] = id_to_oclass.get(int(aid), "") if sym == "O" else ""

    atoms_obj.set_array("role", role)
    atoms_obj.set_array("o_class", o_class)

    cutoffs = elements_info.get("cutoffs", {})
    former_cutoffs = {k: v for k, v in cutoffs.items() if k in formers_set}
    atoms_obj.info["former_cutoffs"] = json.dumps(former_cutoffs)

    return atoms_obj


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

    # 3D structure XYZ - annotate atoms with role and oxygen class for the viewer
    structure_xyz = ""
    try:
        from amorphouspy_api.models import validate_atoms

        atoms_obj = validate_atoms(mq.get("final_structure"))
        if atoms_obj is not None:
            elements_info = structural_analysis.get("elements", {})
            _annotate_atoms_for_viewer(atoms_obj, elements_info)
            structure_xyz = _atoms_to_xyz_string(atoms_obj)
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
