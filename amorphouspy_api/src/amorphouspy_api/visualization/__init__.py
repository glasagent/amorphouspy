"""Visualization helpers for the amorphouspy API.

Each module provides ``prepare_*_plots()`` entry points that accept raw
result dicts and return JSON-encoded Plotly figure dicts ready for the
front-end.
"""

from amorphouspy_api.visualization.cte import prepare_cte_plots
from amorphouspy_api.visualization.elastic import prepare_elastic_plots
from amorphouspy_api.visualization.meltquench import build_temperature_time_plot, prepare_timing_context
from amorphouspy_api.visualization.structure import prepare_structure_context
from amorphouspy_api.visualization.viscosity import prepare_viscosity_plots

__all__ = [
    "build_temperature_time_plot",
    "prepare_cte_plots",
    "prepare_elastic_plots",
    "prepare_structure_context",
    "prepare_timing_context",
    "prepare_viscosity_plots",
]
