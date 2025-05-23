import pytest
from pyiron_glass.mass import get_atomic_mass


def test_get_atomic_mass():
    """Test that the get_atomic_mass function works correctly."""
    # Test by element symbol
    assert get_atomic_mass("H") == pytest.approx(1.008)
    assert get_atomic_mass("O") == pytest.approx(15.999)
    assert get_atomic_mass("Si") == pytest.approx(28.085)
    assert get_atomic_mass("Fe") == pytest.approx(55.845)

    # Test by atomic number
    assert get_atomic_mass(1) == pytest.approx(1.008)  # H
    assert get_atomic_mass(8) == pytest.approx(15.999)  # O
    assert get_atomic_mass(14) == pytest.approx(28.085)  # Si
    assert get_atomic_mass(26) == pytest.approx(55.845)  # Fe
