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

DEFAULT_PRE_EQUILIBRATION_SEED = 48279


def pre_equilibration_fix_override(melt_temperature: float, seed: int = DEFAULT_PRE_EQUILIBRATION_SEED) -> str:
    """Return the input-control ``fix`` override for a standalone pre-equilibration stage.

    Passing ``{"fix": pre_equilibration_fix_override(T)}`` as the runner's
    ``input_control_file`` replaces the generated integrator fix with the
    Langevin + nve/limit pair, so the block runs as its own MD stage with a
    clean potential Config. Requires the runner call to use ``ensemble="nvt"``
    (exactly one generated fix line to replace).

    Args:
        melt_temperature: Target temperature (K) for the Langevin thermostat.
        seed: Seed of the Langevin random forces; LAMMPS requires a positive integer.

    Returns:
        Value for the ``"fix"`` key of ``input_control_file``.

    Raises:
        ValueError: If ``seed`` is not a positive integer.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
        msg = f"Langevin seed must be a positive integer, got {seed!r}."
        raise ValueError(msg)
    return (
        f"langevinnve all langevin {melt_temperature:g} {melt_temperature:g} 0.01 {seed}\n"
        "fix ensemblenve all nve/limit 0.5"
    )
