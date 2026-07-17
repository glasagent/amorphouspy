"""High-temperature pre-equilibration block for melt-quench protocols.

Randomly placed structures contain overlapping atoms; running a short
Langevin + ``nve/limit`` stage before the first protocol stage prevents the
system from exploding ("Lost atoms"). When ``MeltQuenchParams.pre_equilibrate``
is enabled, most melt-quench protocols append this block to the potential
Config of their first stage; the SHIK protocol instead runs it as its own
stage 0 via the runner's input-control fix override.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def melt_block_lines(melt_temperature: float) -> list[str]:
    """Return the Langevin + nve/limit pre-equilibration config lines.

    Args:
        melt_temperature: Target temperature (K) for the Langevin thermostat.

    Returns:
        LAMMPS config lines run once before the first heating stage.
    """
    return [
        f"\nfix langevinnve all langevin {melt_temperature:g} {melt_temperature:g} 0.01 48279\n",
        "\nfix ensemblenve all nve/limit 0.5\n",
        "\nrun 10000\n",
        "\nunfix langevinnve\n",
        "\nunfix ensemblenve\n",
    ]


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


def append_melt_block(potential: pd.DataFrame, melt_temperature: float) -> pd.DataFrame:
    """Return a copy of *potential* with the pre-equilibration block appended to its Config.

    The block lands at the end of the Config, so LAMMPS runs it right after
    the pair setup and before the stage's own fixes.

    Args:
        potential: Single-row potential DataFrame with a ``Config`` column.
        melt_temperature: Target temperature (K) for the Langevin thermostat.

    Returns:
        Copy of the DataFrame with the block appended; the input is unchanged.
    """
    appended = potential.copy()
    appended["Config"] = appended["Config"].apply(lambda lines: lines + melt_block_lines(melt_temperature))
    return appended
