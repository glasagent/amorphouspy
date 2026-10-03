"""Tests for pure functions in amorphouspy.properties.vibrational."""

import itertools

import numpy as np
import pytest
from amorphouspy.properties.vibrational import convert_frequency

UNITS = ["THz", "cm-1", "meV", "rad/ps"]

# ---------------------------------------------------------------------------
# convert_frequency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("unit", "expected"), [("cm-1", 33.35641), ("meV", 4.135668), ("rad/ps", 6.283185)])
def test_convert_frequency_reference_values(unit, expected):
    """1 THz matches the reference value in each unit."""
    assert convert_frequency(1.0, "THz", unit) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(("from_unit", "to_unit"), list(itertools.product(UNITS, UNITS)))
def test_convert_frequency_round_trip(from_unit, to_unit):
    """Converting there and back returns the input for every unit pair."""
    values = np.array([-3.7, 0.0, 0.25, 12.5, 40.0])
    back = convert_frequency(convert_frequency(values, from_unit, to_unit), to_unit, from_unit)
    np.testing.assert_allclose(back, values, rtol=1e-12)
    scalar_back = convert_frequency(convert_frequency(12.5, from_unit, to_unit), to_unit, from_unit)
    assert scalar_back == pytest.approx(12.5, rel=1e-12)


def test_convert_frequency_array_keeps_shape_and_float64():
    """Array input returns a float64 array of the same shape."""
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = convert_frequency(values, "THz", "meV")
    assert result.shape == (2, 3)
    assert result.dtype == np.float64


def test_convert_frequency_integer_input():
    """Integer scalar returns float; integer array returns float64."""
    scalar = convert_frequency(2, "THz", "cm-1")
    assert type(scalar) is float
    array = convert_frequency(np.array([1, 2, 3]), "THz", "cm-1")
    assert array.dtype == np.float64


def test_convert_frequency_negative_sign_preserving():
    """Negative frequencies (imaginary modes) keep their sign."""
    assert convert_frequency(-1.0, "THz", "cm-1") == pytest.approx(-33.35641, rel=1e-6)
    result = convert_frequency(np.array([-2.0, 2.0]), "THz", "meV")
    assert result[0] == pytest.approx(-result[1])
    assert result[0] < 0


def test_convert_frequency_input_unchanged():
    """The input array is not modified in place."""
    values = np.array([-1.0, 1.0, 5.0])
    original = values.copy()
    convert_frequency(values, "THz", "rad/ps")
    np.testing.assert_array_equal(values, original)


def test_convert_frequency_same_unit_returns_copy():
    """from_unit == to_unit returns an equal array that is not the input object."""
    values = np.array([1.0, 2.0])
    result = convert_frequency(values, "meV", "meV")
    np.testing.assert_array_equal(result, values)
    assert result is not values
    assert not np.shares_memory(result, values)


@pytest.mark.parametrize(("from_unit", "to_unit"), [("Hz", "THz"), ("THz", "cm^-1")])
def test_convert_frequency_unknown_unit_raises(from_unit, to_unit):
    """An unknown unit raises ValueError naming the allowed units."""
    with pytest.raises(ValueError, match="allowed units"):
        convert_frequency(1.0, from_unit, to_unit)
