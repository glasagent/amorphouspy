"""Pipeline subpackage of amorphouspy.

High-level workflow functions that compose the lower-level building blocks
(fabrication, LAMMPS, properties) into end-to-end simulation pipelines.

Each pipeline module accepts plain Python arguments (ASE Atoms, dicts, floats)
and returns plain dicts — no web-framework or pydantic dependency.
"""

from amorphouspy.pipelines.meltquench import generate_structure, run_melt_quench
from amorphouspy.pipelines.structural import (
    extract_equilibration_frames,
    run_structural_analysis,
)
from amorphouspy.pipelines.viscosity import run_viscosity_workflow

__all__ = [
    "extract_equilibration_frames",
    "generate_structure",
    "run_melt_quench",
    "run_structural_analysis",
    "run_viscosity_workflow",
]
