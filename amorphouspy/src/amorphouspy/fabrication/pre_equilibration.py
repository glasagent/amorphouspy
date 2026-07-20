"""High-temperature pre-equilibration for melt-quench protocols.

Randomly placed structures contain overlapping atoms; running a short
Langevin + ``nve/limit`` stage before the first protocol stage prevents the
system from exploding ("Lost atoms"). When ``MeltQuenchParams.pre_equilibrate``
is enabled, every melt-quench protocol runs this as its own stage 0 via the
runner's input-control fix override (see
``meltquench_protocols._pre_equilibration_stage``).

Author: Achraf Atila (achraf.atila@bam.de)
"""

from __future__ import annotations


def pre_equilibration_fix_override(melt_temperature: float) -> str:
    """Return the input-control ``fix`` override for a standalone pre-equilibration stage.

    Passing ``{"fix": pre_equilibration_fix_override(T)}`` as the runner's
    ``input_control_file`` replaces the generated integrator fix with the
    Langevin + nve/limit pair, so the block runs as its own MD stage with a
    clean potential Config. Requires the runner call to use ``langevin=False``
    and ``pressure=None`` (exactly one generated fix line to replace).

    Args:
        melt_temperature: Target temperature (K) for the Langevin thermostat.

    Returns:
        Value for the ``"fix"`` key of ``input_control_file``.
    """
    return (
        f"langevinnve all langevin {melt_temperature:g} {melt_temperature:g} 0.01 48279\n"
        "fix ensemblenve all nve/limit 0.5"
    )
