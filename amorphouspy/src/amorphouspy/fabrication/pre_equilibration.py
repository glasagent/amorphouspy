"""High-temperature pre-equilibration block for melt-quench protocols.

Randomly placed structures contain overlapping atoms; running a short
Langevin + ``nve/limit`` stage before the first heating ramp prevents the
system from exploding ("Lost atoms"). The melt-quench protocols append this
block to the potential Config of their first stage when
``MeltQuenchParams.pre_equilibrate`` is enabled.

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
