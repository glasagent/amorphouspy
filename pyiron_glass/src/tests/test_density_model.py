"""Tests for density calculation utilities."""

import pytest

from pyiron_glass import get_glass_density_from_model

# Constants
TOLERANCE = 0.01  # g/cm³ acceptable deviation


def test_get_glass_density_from_model() -> None:
    """Test that the get_glass_density_from_model function works correctly."""
    # Valid composition
    assert abs(get_glass_density_from_model("80SiO2-20Na2O") - 2.391) < TOLERANCE

    # Should raise errors
    with pytest.raises(ValueError, match=r"Sum .* exceeds 100%"):
        get_glass_density_from_model("80SiO2-30Na2O")  # Sum > 100%

    with pytest.raises(ValueError, match=r"Trace oxide .* without remainder"):
        get_glass_density_from_model("0.5Ag2O-0.5SiO2")  # Trace oxide without remainder

    with pytest.raises(ValueError, match=r"Unsupported component"):
        get_glass_density_from_model("50FeO-50SiO2")  # Unsupported component
