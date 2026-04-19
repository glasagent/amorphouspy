"""Electrostatics configuration dataclass for LAMMPS potential generation."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ElectrostaticsConfig:
    """Electrostatics treatment for LAMMPS potential generation.

    Controls the Coulomb solver (DSF, Wolf, PPPM, or Ewald) and the associated
    cutoffs. Fields left as ``None`` fall back to each potential's built-in defaults.

    Args:
        method: Coulomb solver. ``"dsf"`` and ``"wolf"`` are short-range damped
            methods that require ``alpha``. ``"pppm"`` and ``"ewald"`` are
            reciprocal-space methods that append a ``kspace_style`` line and
            ignore ``alpha``.
        short_range_cutoff: Pair-potential cutoff in Å (ignored by BJP).
        long_range_cutoff: Coulomb cutoff in Å.
        alpha: Damping parameter (Å⁻¹) for DSF/Wolf. Ignored for PPPM/Ewald.
        kspace_accuracy: Relative accuracy for PPPM/Ewald (e.g. 1e-5).
    """

    method: Literal["dsf", "wolf", "pppm", "ewald"] = "dsf"
    short_range_cutoff: float | None = None
    long_range_cutoff: float | None = None
    alpha: float | None = None
    kspace_accuracy: float = 1e-5
