"""Tests for oxide charge neutrality validation.

Author: (achraf.atila@bam.de)

This script provides tests for oxide charge neutrality validation.
It tests various oxide formulas against the check_neutral_oxide function,
verifying correct identification of charge-neutral compounds and proper
error handling for non-neutral or invalid inputs.

The tests cover:
- Valid neutral oxides (Na₂O, etc.)
- Non-neutral compounds (NaO, AlO, etc.)
- Invalid formula formats
- Compounds with undetermined oxidation states
- Edge cases and special oxides
"""

import pytest

from pyiron_glass import check_neutral_oxide

# Test parameters
VALID_OXIDES = [
    "Na2O",
    "K2O",
    "MgO",
    "CaO",
    "Al2O3",
    "CO2",
    "SiO2",
    "TiO2",
    "P2O5",
    "SO3",
    "H2O",
    "O2",
    "Na2O2",
    "Fe2O3",
    "Cr2O3",
]

NON_NEUTRAL_CASES = [
    ("NaO", "net charge -1"),
    ("AlO", "net charge 1"),
    ("FeO2", "net charge -2"),
    ("LiO2", "net charge -3"),
    ("C2O", "net charge -2"),
    ("Na2O3", "net charge -4"),
]

INVALID_FORMULAS = [
    "H2O!",
    "Na-2O",
    "3Al2O3",
    "abc",
    "CO@2",
    "",
    123,
]

PROBLEMATIC_COMPOUNDS = [
    "XeO3",
    "Fe4C",
    "HeO",
]


@pytest.mark.parametrize("formula", VALID_OXIDES)
def test_valid_neutral_oxides(formula: str) -> None:
    """Test that valid neutral oxides pass without errors."""
    check_neutral_oxide(formula)


@pytest.mark.parametrize(("formula", "expected_error"), NON_NEUTRAL_CASES)
def test_non_neutral_oxides(formula: str, expected_error: str) -> None:
    """Test detection of non-neutral compounds."""
    with pytest.raises(ValueError, match=expected_error):
        check_neutral_oxide(formula)


@pytest.mark.parametrize("formula", INVALID_FORMULAS)
def test_invalid_formats(formula: str) -> None:
    """Test handling of malformed formulas."""
    with pytest.raises(ValueError, match=r"Invalid oxide formula"):
        check_neutral_oxide(formula)


@pytest.mark.parametrize("formula", PROBLEMATIC_COMPOUNDS)
def test_unsupported_compounds(formula: str) -> None:
    """Test compounds with no oxidation state guesses."""
    with pytest.raises(ValueError, match=r"Cannot determine oxidation states"):
        check_neutral_oxide(formula)


def test_special_cases() -> None:
    """Test edge cases and special oxides."""
    # Peroxide (should be neutral)
    check_neutral_oxide("BaO2")

    # Superoxide (should be neutral)
    check_neutral_oxide("RbO2")

    # Mixed valence oxide (should be neutral)
    check_neutral_oxide("Pb3O4")

    # Single element (not oxide)
    with pytest.raises(ValueError, match=r"Not an oxide"):
        check_neutral_oxide("Fe")

    # Non-oxide compound
    with pytest.raises(ValueError, match=r"net charge"):
        check_neutral_oxide("NaCl")
