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
        heating_rate: Heating rate in K/s (optional, default: 1e14)
        cooling_rate: Cooling rate in K/s (optional, default: 1e14)
        n_print: Print interval for simulation output (optional, default: 1000)
    """

    components: List[str] = Field(..., description="List of oxide components (e.g., ['CaO', 'Al2O3', 'SiO2'])")
    values: List[float] = Field(..., description="List of composition values corresponding to components")
    unit: Literal["wt", "mol"] = Field(..., description="Unit type: 'wt' for weight percent or 'mol' for molar percent")
    heating_rate: int = Field(default=int(1e14), description="Heating rate in K/s (default: 1e14)")
    cooling_rate: int = Field(default=int(1e14), description="Cooling rate in K/s (default: 1e14)")
    n_print: int = Field(default=1000, description="Print interval for simulation output (default: 1000)")

    @field_validator("values")
    @classmethod
    def validate_values_sum(cls, v):
        """Ensure composition values sum to approximately 100 (for percentages) or 1 (for fractions)."""
        total = sum(v)
        if total > 1.1:  # Likely percentages
            if not (95 <= total <= 105):
                raise ValueError(f"Composition values sum to {total}, expected around 100 for percentages")
        else:  # Likely fractions
            if not (0.95 <= total <= 1.05):
                raise ValueError(f"Composition values sum to {total}, expected around 1.0 for fractions")
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
    final_structure: str = Field(..., description="String representation of final structure")
    mean_temperature: float = Field(..., description="Mean temperature during final phase (K)")
    final_density: float = Field(..., description="Final calculated density (g/cm³)")
    simulation_steps: int = Field(..., description="Total simulation steps completed")
