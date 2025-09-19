"""Pydantic models for pyiron-glass API requests and responses.

This module contains data validation models for various simulation types
including meltquench simulations and other glass modeling workflows.
"""

from io import StringIO
from typing import Annotated, Literal

from ase import Atoms
from ase.io import read, write
from pydantic import BaseModel, Field, PlainSerializer, PlainValidator, ValidationInfo, field_validator
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
    """Serialize ASE Atoms to JSON string."""
    json_buffer = StringIO()
    write(json_buffer, atoms, format="json")
    return json_buffer.getvalue()


def validate_atoms(v: Atoms | dict | str | None) -> Atoms | None:
    """Validate and convert input to ASE Atoms object."""
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
    Atoms | None, PlainValidator(validate_atoms), PlainSerializer(serialize_atoms, return_type=str, when_used="unless-none")
]


# Export the serialization functions for use in other modules
__all__ = ["AtomsType", "MeltquenchRequest", "MeltquenchResult", "serialize_atoms", "validate_atoms"]


class MeltquenchRequest(BaseModel):
    """Request model for meltquench simulation.

    Attributes:
        components: List of component names (e.g., ["CaO", "Al2O3", "SiO2"])
        values: List of composition values corresponding to components
        unit: Unit type - either "wt" (weight percent) or "mol" (molar percent)
        heating_rate: Heating rate in K/s (optional, default: 1e14)
        cooling_rate: Cooling rate in K/s (optional, default: 1e14)
        n_print: Print interval for simulation output (optional, default: 1000)

    """

    components: list[str] = Field(..., description="List of oxide components (e.g., ['CaO', 'Al2O3', 'SiO2'])")
    values: list[float] = Field(..., description="List of composition values corresponding to components")
    unit: Literal["wt", "mol"] = Field(..., description="Unit type: 'wt' for weight percent or 'mol' for molar percent")
    heating_rate: int = Field(default=int(1e14), description="Heating rate in K/s (default: 1e14)")
    cooling_rate: int = Field(default=int(1e14), description="Cooling rate in K/s (default: 1e14)")
    n_print: int = Field(default=1000, description="Print interval for simulation output (default: 1000)")

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
        composition: Final composition string used in simulation
        final_structure: ASE Atoms object representing the final atomic structure
        mean_temperature: Average temperature during final equilibration
        simulation_steps: Total number of simulation steps completed
        structural_analysis: Structural analysis results from glass structure analysis

    """

    composition: str = Field(..., description="Composition string used in simulation")
    final_structure: AtomsType = Field(..., description="ASE Atoms object of final structure")
    mean_temperature: float = Field(..., description="Mean temperature during final phase (K)")
    simulation_steps: int = Field(..., description="Total simulation steps completed")
    structural_analysis: StructureData | dict = Field(..., description="Structural analysis results")
