"""Tests for amorphouspy.shared utility functions."""

import numpy as np
import pytest
from amorphouspy.shared import count_distribution, get_element_types_dict, running_mean, type_to_dict


def test_get_element_types_dict_basic() -> None:
    """Elements are sorted alphabetically and assigned 1-based integer types."""
    atoms = [
        {"element": "Si"},
        {"element": "O"},
        {"element": "O"},
        {"element": "Na"},
    ]
    result = get_element_types_dict(atoms)
    # Alphabetical: Na=1, O=2, Si=3
    assert result == {"Na": 1, "O": 2, "Si": 3}


def test_get_element_types_dict_single_element() -> None:
    """Single-element system returns type 1."""
    atoms = [{"element": "Si"}, {"element": "Si"}]
    result = get_element_types_dict(atoms)
    assert result == {"Si": 1}


def test_count_distribution_basic() -> None:
    """Coordination numbers are bucketed into a frequency histogram."""
    coord_numbers = {1: 4, 2: 4, 3: 3, 4: 3}
    result = count_distribution(coord_numbers)
    assert result == {4: 2, 3: 2}


def test_count_distribution_uniform() -> None:
    """All atoms with the same coordination → single entry."""
    coord_numbers = {1: 4, 2: 4, 3: 4}
    result = count_distribution(coord_numbers)
    assert result == {4: 3}


def test_count_distribution_empty() -> None:
    """Empty input returns empty dict."""
    assert count_distribution({}) == {}


def test_type_to_dict_basic() -> None:
    """Atomic numbers map to correct element symbols."""
    result = type_to_dict(np.array([8, 14]))
    assert result == {8: "O", 14: "Si"}


def test_type_to_dict_with_duplicates() -> None:
    """Duplicate atomic numbers are deduplicated."""
    result = type_to_dict(np.array([8, 8, 14, 14, 11]))
    assert result == {8: "O", 11: "Na", 14: "Si"}


def test_type_to_dict_single() -> None:
    """Single type returns single-entry dict."""
    result = type_to_dict(np.array([14]))
    assert result == {14: "Si"}


def test_running_mean_n1_returns_unchanged() -> None:
    """N=1 returns the original data unchanged."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = running_mean(data, n=1)
    np.testing.assert_array_equal(result, data)


def test_running_mean_basic() -> None:
    """N=3 on constant data returns all 1s in the valid window."""
    data = np.ones(7)
    result = running_mean(data, n=3)
    # pad_left=1, pad_right=1 → indices [1:-1] are valid
    np.testing.assert_array_almost_equal(result[1:-1], np.ones(5))


def test_running_mean_nan_edges() -> None:
    """Edges outside the valid window are NaN."""
    data = np.ones(6)
    result = running_mean(data, n=3)
    assert np.isnan(result[0])
    assert np.isnan(result[-1])


def test_running_mean_shape() -> None:
    """Output has the same shape as the input."""
    data = np.arange(10, dtype=float)
    result = running_mean(data, n=3)
    assert result.shape == data.shape


def test_running_mean_correctness() -> None:
    """Spot-check values against manual calculation."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = running_mean(data, n=3)
    # Valid range: indices 1 to 3
    assert result[1] == pytest.approx(2.0)
    assert result[2] == pytest.approx(3.0)
    assert result[3] == pytest.approx(4.0)
