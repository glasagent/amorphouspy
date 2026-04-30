"""Electrostatics configuration classes for LAMMPS potential generation.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from typing import Literal

from pydantic import BaseModel, Field


class InteractionConfig(BaseModel):
    """Base class for LAMMPS Coulomb solver settings."""

    def kspace_line(self) -> str | None:
        return None


class DsfConfig(InteractionConfig):
    """Damped Shifted Force electrostatics.

    Emitted as ``coul/dsf <alpha> <long_range_cutoff>``.
    """

    lammps_keyword: Literal["dsf"] = "dsf"
    long_range_cutoff: float | None = Field(default=None, description="Coulomb cutoff in Å", gt=0)
    alpha: float | None = Field(default=None, description="Damping parameter (Å⁻¹)", gt=0)

    def coul_pair_style(self, cutoff: float, alpha: float | None = None) -> str:
        if alpha is None:
            msg = "alpha is required for DSF electrostatics"
            raise ValueError(msg)
        return f"coul/{self.lammps_keyword} {alpha} {cutoff}"


class WolfConfig(InteractionConfig):
    """Wolf summation electrostatics.

    Emitted as ``coul/wolf <alpha> <long_range_cutoff>``.
    """

    lammps_keyword: Literal["wolf"] = "wolf"
    long_range_cutoff: float | None = Field(default=None, description="Coulomb cutoff in Å", gt=0)
    alpha: float | None = Field(default=None, description="Damping parameter (Å⁻¹)", gt=0)

    def coul_pair_style(self, cutoff: float, alpha: float | None = None) -> str:
        if alpha is None:
            msg = "alpha is required for Wolf electrostatics"
            raise ValueError(msg)
        return f"coul/{self.lammps_keyword} {alpha} {cutoff}"


class PppmConfig(InteractionConfig):
    """PPPM (Particle-Particle Particle-Mesh) electrostatics.

    Emits ``coul/long <long_range_cutoff>`` plus a ``kspace_style pppm`` line.
    """

    lammps_keyword: Literal["pppm"] = "pppm"
    long_range_cutoff: float | None = Field(default=None, description="Coulomb cutoff in Å", gt=0)
    kspace_accuracy: float = Field(default=1e-5, description="Relative accuracy for the k-space sum", gt=0, le=1)

    def coul_pair_style(self, cutoff: float) -> str:
        return f"coul/long {cutoff}"

    def kspace_line(self) -> str:
        return f"kspace_style {self.lammps_keyword} {self.kspace_accuracy}"


class EwaldConfig(InteractionConfig):
    """Ewald summation electrostatics.

    Emits ``coul/long <long_range_cutoff>`` plus a ``kspace_style ewald`` line.
    """

    lammps_keyword: Literal["ewald"] = "ewald"
    long_range_cutoff: float | None = Field(default=None, description="Coulomb cutoff in Å", gt=0)
    kspace_accuracy: float = Field(default=1e-5, description="Relative accuracy for the k-space sum", gt=0, le=1)

    def coul_pair_style(self, cutoff: float) -> str:
        return f"coul/long {cutoff}"

    def kspace_line(self) -> str:
        return f"kspace_style {self.lammps_keyword} {self.kspace_accuracy}"
