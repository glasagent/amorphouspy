"""Pipeline subpackage of amorphouspy.

High-level workflow functions that compose the lower-level building blocks
(fabrication, LAMMPS, properties) into end-to-end simulation pipelines.

Each pipeline module accepts plain Python arguments (ASE Atoms, dicts, floats)
and returns plain dicts — no web-framework or pydantic dependency.
"""

from amorphouspy.pipelines.viscosity import submit_viscosity_workflow

# Analyses that build their own executor sub-DAG rather than running as a
# single submitted function.  Maps analysis name → DAG-building callable.
PIPELINE_ANALYSES: dict[str, object] = {
    "viscosity": submit_viscosity_workflow,
}
