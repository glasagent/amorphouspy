"""Tests for oxide charge neutrality validation."""

import pytest

from pyiron_glass.structure import check_neutral_oxide


def test_check_neutral_oxide_na2o() -> None:
    """Test that Na2O is correctly identified as charge neutral."""
    check_neutral_oxide("Na2O")


def test_check_neutral_oxide_al2o3() -> None:
    """Test that Al2O3 is correctly identified as charge neutral."""
    check_neutral_oxide("Al2O3")


def test_check_neutral_oxide_nao2() -> None:
    """Test that NaO2 is correctly identified as NOT charge neutral."""
    with pytest.raises(ValueError, match="Cannot determine oxidation states for 'NaO2'") as excinfo:
        check_neutral_oxide("NaO2")
    assert "Cannot determine oxidation states" in str(excinfo.value)


def test_check_neutral_oxide_no_oxidation_states() -> None:
    """Test that formulas without oxidation state guesses raise appropriate errors."""
    with pytest.raises(ValueError, match="'ABC' is not a valid Element") as excinfo:
        check_neutral_oxide("ABC")
    assert "is not a valid Element" in str(excinfo.value)
