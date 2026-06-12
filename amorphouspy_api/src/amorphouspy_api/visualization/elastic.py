"""Elastic moduli visualization helpers (bar chart, Cij heatmap)."""

from __future__ import annotations

from typing import Any


def _build_elastic_moduli_plot(moduli: dict[str, float]) -> dict:
    """Bar chart of isotropic elastic moduli (B, G, E)."""
    labels = []
    values = []
    for key, label in [("B", "Bulk (B)"), ("G", "Shear (G)"), ("E", "Young's (E)")]:
        if key in moduli:
            labels.append(label)
            values.append(moduli[key])

    return {
        "data": [
            {
                "x": labels,
                "y": values,
                "type": "bar",
                "marker": {"color": ["#1f77b4", "#ff7f0e", "#2ca02c"]},
            }
        ],
        "layout": {
            "title": "Isotropic Elastic Moduli (VRH Average)",
            "xaxis": {"title": ""},
            "yaxis": {"title": "GPa"},
            "template": "plotly_white",
        },
    }


def _build_cij_heatmap(cij: list[list[float]]) -> dict:
    """Heatmap of the 6x6 Cij stiffness tensor."""
    labels = ["C1", "C2", "C3", "C4", "C5", "C6"]
    return {
        "data": [
            {
                "z": cij,
                "x": labels,
                "y": labels,
                "type": "heatmap",
                "colorscale": "RdBu",
                "zmid": 0,
                "colorbar": {"title": {"text": "GPa", "side": "right"}},
            }
        ],
        "layout": {
            "title": "Stiffness Tensor Cij (Voigt Notation)",
            "xaxis": {"title": "j", "dtick": 1},
            "yaxis": {"title": "i", "dtick": 1, "autorange": "reversed"},
            "template": "plotly_white",
            "width": 500,
            "height": 500,
        },
    }


def prepare_elastic_plots(elastic_data: dict[str, Any]) -> dict[str, str]:
    """Build JSON-encoded Plotly plots from elastic result data.

    Returns:
        Dict with keys ``moduli`` and optionally ``cij_heatmap``.
    """
    import json

    plots: dict[str, str] = {}

    moduli = elastic_data.get("moduli")
    if moduli:
        plots["moduli"] = json.dumps(_build_elastic_moduli_plot(moduli))

    cij = elastic_data.get("Cij")
    if cij:
        plots["cij_heatmap"] = json.dumps(_build_cij_heatmap(cij))

    return plots
