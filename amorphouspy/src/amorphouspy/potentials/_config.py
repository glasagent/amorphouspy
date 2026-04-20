"""Electrostatics configuration classes for LAMMPS potential generation.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InteractionConfig(BaseModel):
    """Base class for LAMMPS Coulomb solver settings."""

    model_config = ConfigDict(frozen=True)


class DsfConfig(InteractionConfig):
    """Damped Shifted Force electrostatics.

    Emitted as ``coul/dsf <alpha> <long_range_cutoff>``.
    """

    method: Literal["dsf"] = "dsf"
    long_range_cutoff: float | None = Field(default=None, description="Coulomb cutoff in Å", gt=0)
    alpha: float | None = Field(default=None, description="Damping parameter (Å⁻¹)", gt=0)


class WolfConfig(InteractionConfig):
    """Wolf summation electrostatics.

    Emitted as ``coul/wolf <alpha> <long_range_cutoff>``.
    """

    method: Literal["wolf"] = "wolf"
    long_range_cutoff: float | None = Field(default=None, description="Coulomb cutoff in Å", gt=0)
    alpha: float | None = Field(default=None, description="Damping parameter (Å⁻¹)", gt=0)


class PppmConfig(InteractionConfig):
    """PPPM (Particle-Particle Particle-Mesh) electrostatics.

    Emits ``coul/long <long_range_cutoff>`` plus a ``kspace_style pppm`` line.
    """

    method: Literal["pppm"] = "pppm"
    long_range_cutoff: float | None = Field(default=None, description="Coulomb cutoff in Å", gt=0)
    kspace_accuracy: float = Field(default=1e-5, description="Relative accuracy for the k-space sum", gt=0, le=1)


class EwaldConfig(InteractionConfig):
    """Ewald summation electrostatics.

    Emits ``coul/long <long_range_cutoff>`` plus a ``kspace_style ewald`` line.
    """

    method: Literal["ewald"] = "ewald"
    long_range_cutoff: float | None = Field(default=None, description="Coulomb cutoff in Å", gt=0)
    kspace_accuracy: float = Field(default=1e-5, description="Relative accuracy for the k-space sum", gt=0, le=1)
