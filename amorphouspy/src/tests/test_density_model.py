"""Tests for density calculation utilities."""

from amorphouspy.structure import get_glass_density_from_model

# Constants
TOLERANCE = 0.01  # g/cm³ acceptable deviation


def test_get_glass_density_from_model() -> None:
    """Test density calculation with various cases."""
    density = get_glass_density_from_model({"SiO2": 80, "Na2O": 20})
    assert abs(density - 2.391) < TOLERANCE

    assert density > 0, "Density should be positive"
