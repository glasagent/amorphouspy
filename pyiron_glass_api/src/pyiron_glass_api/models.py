"""Pydantic models for pyiron-glass API requests and responses.

This module contains data validation models for various simulation types
including meltquench simulations and other glass modeling workflows.
"""

from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator

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
FRACTION_MIN = 0.95
FRACTION_MAX = 1.05

# Error messages
PERCENTAGE_ERROR_MSG = "Composition values sum to {total}, expected around 100 for percentages"
FRACTION_ERROR_MSG = "Composition values sum to {total}, expected around 1.0 for fractions"
LENGTH_ERROR_MSG = "Components and values lists must have the same length"


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


class StructuralAnalysis(BaseModel):
    """Structural analysis results from glass structure analysis.

    Attributes:
        density: Calculated density of the structure (g/cm³)
        network_connectivity: Network connectivity value (0-1)
        mean_coordination: Dictionary of mean coordination numbers for different atom types
        qn_distribution: Q^n distribution for network formers
        network_formers: List of network forming elements
        modifiers: List of modifier elements
        error: Error message if analysis failed (optional)
    """

    density: float | None = Field(None, description="Calculated density (g/cm³)")
    network_connectivity: float | None = Field(None, description="Network connectivity (0-1)")
    mean_coordination: dict[str, float] = Field(default_factory=dict, description="Mean coordination numbers")
    qn_distribution: dict[str, float] = Field(default_factory=dict, description="Q^n distribution")
    network_formers: list[str] = Field(default_factory=list, description="Network forming elements")
    modifiers: list[str] = Field(default_factory=list, description="Modifier elements")
    error: str | None = Field(None, description="Error message if analysis failed")


class MeltquenchResult(BaseModel):
    """Result model for completed meltquench simulation.

    Attributes:
        composition: Final composition string used in simulation
        final_structure: String representation of the final atomic structure
        mean_temperature: Average temperature during final equilibration
        simulation_steps: Total number of simulation steps completed
        structural_analysis: Structural analysis results from glass structure analysis

    """

    composition: str = Field(..., description="Composition string used in simulation")
    final_structure: str = Field(..., description="String representation of final structure")
    mean_temperature: float = Field(..., description="Mean temperature during final phase (K)")
    simulation_steps: int = Field(..., description="Total simulation steps completed")
    structural_analysis: StructuralAnalysis = Field(..., description="Structural analysis results")
