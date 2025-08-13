"""Tests for oxide charge neutrality validation."""

import pytest
from pyiron_glass.structure import check_neutral_oxide  # Replace 'your_module' with the actual module name

def test_check_neutral_oxide_na2o():
    """Test that Na2O is correctly identified as charge neutral."""
    check_neutral_oxide("Na2O")  # Should not raise any exception

def test_check_neutral_oxide_al2o3():
    """Test that Al2O3 is correctly identified as charge neutral."""
    check_neutral_oxide("Al2O3")  # Should not raise any exception

def test_check_neutral_oxide_nao2():
    """Test that NaO2 is correctly identified as NOT charge neutral."""
    with pytest.raises(ValueError) as excinfo:
        check_neutral_oxide("NaO2")
    assert "Cannot determine oxidation states" in str(excinfo.value)


def test_check_neutral_oxide_no_oxidation_states():
    """Test that formulas without oxidation state guesses raise appropriate errors."""
    # Note: This test might need adjustment based on pymatgen's behavior with unusual formulas
    with pytest.raises(ValueError) as excinfo:
        check_neutral_oxide("XyZ")  # Assuming this is a formula that pymatgen can't determine oxidation states for
    assert "is not a valid Element" in str(excinfo.value)