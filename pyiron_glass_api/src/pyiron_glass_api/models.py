"""Pydantic models for pyiron-glass API requests and responses.

This module contains data validation models for various simulation types
including meltquench simulations and other glass modeling workflows.
"""

from io import StringIO
from typing import Annotated, Literal

from ase import Atoms
from ase.io import read, write
from pydantic import BaseModel, Field, PlainSerializer, PlainValidator, ValidationInfo, field_validator, model_validator
from pyiron_glass.workflows.structural_analysis import StructureData

# Constants for composition validation
PERCENTAGE_THRESHOLD = 1.1
PERCENTAGE_MIN = 95
PERCENTAGE_MAX = 105
FRACTION_MIN = 0.95
FRACTION_MAX = 1.05

# Error messages
PERCENTAGE_ERROR_MSG = "Composition values sum to {total}, expected around 100 for percentages"
FRACTION_ERROR_MSG = "Composition values sum to {total}, expected around 1.0 for fractions"
LENGTH_ERROR_MSG = "Components and values lists must have the same length"


def serialize_atoms(atoms: Atoms) -> str:
    """Serialize ASE Atoms to JSON string.

    Args:
        atoms: The ASE Atoms object to serialize.

    Returns:
        JSON string representation of the Atoms object.
    """
    json_buffer = StringIO()
    write(json_buffer, atoms, format="json")
    return json_buffer.getvalue()


def validate_atoms(v: Atoms | dict | str | None) -> Atoms | None:
    """Validate and convert input to ASE Atoms object.

    Args:
        v: Input value which can be an ASE Atoms object, a dictionary, a JSON string, or None.

    Returns:
        The validated ASE Atoms object or None if input is None.

    Raises:
        ValueError: If reconstruction from dict or parsing from string fails.
        TypeError: If input type is not supported.
    """
    if v is None:
        return None
    elif isinstance(v, Atoms):
        return v
    elif isinstance(v, dict):
        # Try to reconstruct from dict
        try:
            return Atoms(**v)
        except Exception as e:
            msg = f"Could not reconstruct Atoms from dict: {e}"
            raise ValueError(msg) from e
    elif isinstance(v, str):
        # Try to parse from JSON string format
        try:
            return read(StringIO(v), format="json")
        except Exception as e:
            msg = f"Could not parse Atoms from string: {e}"
            raise ValueError(msg) from e
    else:
        msg = f"final_structure must be ASE Atoms object, dict, string, or None, got {type(v)}"
        raise TypeError(msg)


# Custom type for ASE Atoms with serialization (allowing None)
AtomsType = Annotated[
    Atoms | None,
    PlainValidator(validate_atoms),
    PlainSerializer(serialize_atoms, return_type=str, when_used="unless-none"),
]


# Export the serialization functions for use in other modules
__all__ = [
    "AtomsType",
    "MeltquenchRequest",
    "MeltquenchResult",
    "ViscosityRequest",
    "ViscosityResult",
    "serialize_atoms",
    "validate_atoms",
]


class MeltquenchRequest(BaseModel):
    """Request model for meltquench simulation.

    Attributes:
        components: List of oxide components (e.g., ["CaO", "Al2O3", "SiO2"]).
        values: List of composition values corresponding to components.
        unit: Unit type - either "wt" (weight percent) or "mol" (molar percent).
        heating_rate: Heating rate in K/s (default: 1e14).
        cooling_rate: Cooling rate in K/s (default: 1e12).
        n_print: Print interval for simulation output (default: 1000).
        n_atoms: Target number of atoms for the generated structure (default: 5000).
        potential_type: Type of interatomic potential to use (default: 'pmmcs').
    """

    components: list[str] = Field(..., description="List of oxide components (e.g., ['CaO', 'Al2O3', 'SiO2'])")
    values: list[float] = Field(..., description="List of composition values corresponding to components")
    unit: Literal["wt", "mol"] = Field(..., description="Unit type: 'wt' for weight percent or 'mol' for molar percent")
    heating_rate: int = Field(default=int(1e14), description="Heating rate in K/s (default: 100K/ps)")
    cooling_rate: int = Field(default=int(1e12), description="Cooling rate in K/s (default: 1K/ps)")
    n_print: int = Field(default=1000, description="Print interval for simulation output (default: 1000)")
    n_atoms: int = Field(default=5000, description="Target number of atoms for the generated structure (default: 5000)")
    potential_type: Literal["shik", "bjp", "pmmcs"] = Field(
        default="pmmcs", description="Type of interatomic potential to use (default: 'pmmcs')"
    )

    @field_validator("values")
    @classmethod
    def validate_values_sum(cls, v: list[float]) -> list[float]:
        """Ensure composition values sum to approximately 100 (for percentages) or 1 (for fractions)."""
        total = sum(v)
        if total > PERCENTAGE_THRESHOLD:  # Likely percentages
            if not (PERCENTAGE_MIN <= total <= PERCENTAGE_MAX):
                msg = PERCENTAGE_ERROR_MSG.format(total=total)
                raise ValueError(msg)
        elif not (FRACTION_MIN <= total <= FRACTION_MAX):
            msg = FRACTION_ERROR_MSG.format(total=total)
            raise ValueError(msg)
        return v

    @field_validator("components")
    @classmethod
    def validate_components_length(cls, v: list[str], info: ValidationInfo) -> list[str]:
        """Ensure components and values lists have the same length."""
        if info.data and "values" in info.data and len(v) != len(info.data["values"]):
            raise ValueError(LENGTH_ERROR_MSG)
        return v


class MeltquenchResult(BaseModel):
    """Result model for completed meltquench simulation.

    Attributes:
        composition: Composition string used in simulation.
        final_structure: ASE Atoms object representing the final atomic structure.
        mean_temperature: Mean temperature during final phase (K).
        simulation_steps: Total number of simulation steps completed.
        structural_analysis: Structural analysis results from glass structure analysis.
    """

    composition: str = Field(..., description="Composition string used in simulation")
    final_structure: AtomsType = Field(..., description="ASE Atoms object of final structure")
    mean_temperature: float = Field(..., description="Mean temperature during final phase (K)")
    simulation_steps: int = Field(..., description="Total simulation steps completed")
    structural_analysis: StructureData | dict = Field(..., description="Structural analysis results")


class ViscosityRequest(BaseModel):
    """Request model for viscosity simulation.

    The simulation can either start from a fresh melt-quench run (via nested MeltquenchRequest)
    or from a user-provided atomic structure.

    Attributes:
        meltquench_request: Optional nested meltquench request describing how to generate the structure.
        initial_structure: Optional initial structure; if provided and meltquench_request is None,
            viscosity is computed starting from this structure.
        temperature_sim: Target simulation temperature in Kelvin for the viscosity run.
        timestep: MD timestep in femtoseconds.
        n_timesteps: Number of MD steps for the viscosity production run.
        n_print: Thermodynamic output frequency.
        potential_type: Type of interatomic potential to use.
        max_lag: Optional maximum lag (number of steps) for the viscosity post-processing.
    """

    meltquench_request: MeltquenchRequest | None = Field(
        default=None,
        description="Optional nested meltquench request used to generate a glass structure before viscosity.",
    )
    initial_structure: AtomsType = Field(
        default=None,
        description="Optional initial structure; if provided, viscosity is computed starting from this structure.",
    )
    temperatures: list[float] | None = Field(
        default=None,
        description="List of simulation temperatures in Kelvin for viscosity calculation. "
        "If omitted, temperature_sim (deprecated) can be used as a single temperature.",
    )
    # Backwards-compatible single-temperature field; internally converted to `temperatures`.
    temperature_sim: float | None = Field(
        default=None,
        description="Single simulation temperature in Kelvin for viscosity calculation "
        "(deprecated, use 'temperatures' instead).",
    )
    timestep: float = Field(default=1.0, description="MD timestep in femtoseconds (default: 1.0)")
    n_timesteps: int = Field(
        default=10_000_000, description="Number of MD steps for the viscosity production run (default: 10_000_000)"
    )
    n_print: int = Field(default=1, description="Thermodynamic output frequency (default: every step)")
    potential_type: Literal["shik", "bjp", "pmmcs"] = Field(
        default="pmmcs", description="Type of interatomic potential to use (default: 'pmmcs')"
    )
    max_lag: int | None = Field(
        default=1_000_000,
        description=(
            "Maximum correlation lag (number of time steps) for Green-Kubo viscosity post-processing; "
            "if None, uses full trajectory length."
        ),
    )

    @model_validator(mode="after")
    def validate_source(self) -> "ViscosityRequest":
        """Normalize temperature fields and ensure a valid source for viscosity."""
        # Normalize temperature specification
        if self.temperatures is None and self.temperature_sim is not None:
            self.temperatures = [self.temperature_sim]
        if not self.temperatures:
            msg = "At least one simulation temperature must be provided via 'temperatures' or 'temperature_sim'"
            raise ValueError(msg)

        # Ensure that at least one of meltquench_request or initial_structure is provided
        if self.meltquench_request is None and self.initial_structure is None:
            msg = "Either 'meltquench_request' or 'initial_structure' must be provided for viscosity simulations"
            raise ValueError(msg)
        return self


class ViscosityResult(BaseModel):
    """Result model for completed viscosity simulation.

    Attributes:
        kind: Discriminator field identifying this as a viscosity result.
        composition: Optional composition string (if derived from meltquench).
        temperature: Mean simulation temperature in Kelvin.
        viscosity: Viscosity in Pa·s computed via the Green-Kubo formalism.
        max_lag: List of cutoff correlation times (per stress component) in picoseconds.
        simulation_steps: Number of MD steps used for the viscosity production run.
    """

    kind: Literal["viscosity"] = Field("viscosity", description="Result type discriminator")
    composition: str | None = Field(
        default=None,
        description="Composition string used in simulation (if generated from meltquench); "
        "None for custom input structures.",
    )
    temperatures: list[float] = Field(..., description="List of simulation temperatures during viscosity runs (K)")
    viscosities: list[float] = Field(
        ...,
        description="List of viscosities computed from Green-Kubo analysis (Pa·s), matching the temperatures list",
    )
    max_lag: list[float] = Field(
        ...,
        description=(
            "List of maximum cutoff correlation times (per temperature) in picoseconds used in viscosity integration"
        ),
    )
    simulation_steps: list[int] = Field(
        ..., description="List of MD steps used for viscosity production runs at each temperature"
    )
    lag_times_ps: list[list[float]] = Field(
        default_factory=list,
        description="List of lag time arrays (per temperature) in picoseconds for plotting SACF and running viscosity",
    )
    sacf_data: list[list[float]] = Field(
        default_factory=list,
        description="List of averaged normalized SACF arrays (per temperature)",
    )
    viscosity_running: list[list[float]] = Field(
        default_factory=list,
        description="List of running viscosity arrays (per temperature) showing convergence vs lag time (Pa·s)",
    )
