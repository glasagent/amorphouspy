"""
Pydantic models for pyiron-glass API requests and responses.

This module contains data validation models for various simulation types
including meltquench simulations and other glass modeling workflows.
"""

from typing import List, Literal
from pydantic import BaseModel, Field, field_validator


class MeltquenchRequest(BaseModel):
    """
    Request model for meltquench simulation.

    Attributes:
        components: List of component names (e.g., ["CaO", "Al2O3", "SiO2"])
        values: List of composition values corresponding to components
        unit: Unit type - either "wt" (weight percent) or "mol" (molar percent)
        n_molecules: Number of molecules for the simulation (default: 200)
        density: Target density in g/cm³ (default: 2.69)
        temperature_high: High temperature for melting in K (default: 5000)
        temperature_low: Low temperature for quenching in K (default: 300)
    """

    components: List[str] = Field(
        ..., description="List of oxide components (e.g., ['CaO', 'Al2O3', 'SiO2'])"
    )
    values: List[float] = Field(
        ..., description="List of composition values corresponding to components"
    )
    unit: Literal["wt", "mol"] = Field(
        ..., description="Unit type: 'wt' for weight percent or 'mol' for molar percent"
    )
    n_molecules: int = Field(
        200, description="Number of molecules for simulation", gt=0
    )
    density: float = Field(2.69, description="Target density in g/cm³", gt=0)
    temperature_high: int = Field(
        5000, description="High temperature for melting in K", gt=0
    )
    temperature_low: int = Field(
        300, description="Low temperature for quenching in K", gt=0
    )

    @field_validator("values")
    @classmethod
    def validate_values_sum(cls, v):
        """Ensure composition values sum to approximately 100 (for percentages) or 1 (for fractions)."""
        total = sum(v)
        if total > 1.1:  # Likely percentages
            if not (95 <= total <= 105):
                raise ValueError(
                    f"Composition values sum to {total}, expected around 100 for percentages"
                )
        else:  # Likely fractions
            if not (0.95 <= total <= 1.05):
                raise ValueError(
                    f"Composition values sum to {total}, expected around 1.0 for fractions"
                )
        return v

    @field_validator("components")
    @classmethod
    def validate_components_length(cls, v, info):
        """Ensure components and values lists have the same length."""
        if info.data and "values" in info.data and len(v) != len(info.data["values"]):
            raise ValueError("Components and values lists must have the same length")
        return v


class MeltquenchResult(BaseModel):
    """
    Result model for completed meltquench simulation.

    Attributes:
        composition: Final composition string used in simulation
        final_structure: String representation of the final atomic structure
        mean_temperature: Average temperature during final equilibration
        final_density: Calculated density of the final structure (g/cm³)
        simulation_steps: Total number of simulation steps completed
    """

    composition: str = Field(..., description="Composition string used in simulation")
    final_structure: str = Field(
        ..., description="String representation of final structure"
    )
    mean_temperature: float = Field(
        ..., description="Mean temperature during final phase (K)"
    )
    final_density: float = Field(..., description="Final calculated density (g/cm³)")
    simulation_steps: int = Field(..., description="Total simulation steps completed")
