"""Shared melt pre-equilibration block for LAMMPS potential configs.

Single source of truth for the Langevin + nve/limit block that potential
generators append when ``melt=True``, and for stripping it again before
stages that must not re-melt the structure.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Temperature-independent markers identifying the melt block lines, so the
# same strip works for every potential regardless of its melt temperature.
_MELT_BLOCK_MARKERS = (
    "fix langevinnve all langevin",
    "fix ensemblenve all nve/limit",
    "run 10000",
    "unfix langevinnve",
    "unfix ensemblenve",
)


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


def has_melt_block(potential: pd.DataFrame) -> bool:
    """Return True if the potential Config contains a melt pre-equilibration block."""
    return any(any(marker in line for marker in _MELT_BLOCK_MARKERS) for line in potential.loc[0, "Config"])


def strip_melt_block(potential: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *potential* with the melt pre-equilibration block removed.

    Idempotent and a no-op for configs generated with ``melt=False``.

    Args:
        potential: Single-row potential DataFrame with a ``Config`` column.

    Returns:
        Copy of the DataFrame without the melt block lines.
    """
    stripped = potential.copy()
    stripped["Config"] = stripped["Config"].apply(
        lambda lines: [line for line in lines if not any(marker in line for marker in _MELT_BLOCK_MARKERS)]
    )
    return stripped


def set_melt_block_temperature(potential: pd.DataFrame, melt_temperature: float) -> pd.DataFrame:
    """Return a copy of *potential* with its melt block retuned to *melt_temperature*.

    The block sits at the end of the Config, so stripping and re-appending
    preserves the line order. Potentials without a melt block are returned
    unchanged -- no block is injected.

    Args:
        potential: Single-row potential DataFrame with a ``Config`` column.
        melt_temperature: Target temperature (K) for the Langevin thermostat.

    Returns:
        Copy of the DataFrame with the melt block at the requested temperature.
    """
    if not has_melt_block(potential):
        return potential

    retuned = strip_melt_block(potential)
    retuned["Config"] = retuned["Config"].apply(lambda lines: lines + melt_block_lines(melt_temperature))
    return retuned
