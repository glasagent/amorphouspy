"""Tests for helper functions in amorphouspy.pipelines.viscosity."""

from __future__ import annotations

import itertools

import pytest
from amorphouspy.atoms.shared import _logspace, downsample_log

# ---------------------------------------------------------------------------
# _logspace
# ---------------------------------------------------------------------------


class TestLogspace:
    """Tests for the _logspace helper."""

    def test_empty_when_num_zero(self) -> None:
        """Return empty list when num is zero."""
        assert _logspace(0, 100, 0) == []

    def test_empty_when_num_negative(self) -> None:
        """Return empty list when num is negative."""
        assert _logspace(0, 100, -1) == []

    def test_single_returns_stop(self) -> None:
        """Single-element result equals stop value."""
        result = _logspace(0, 99, 1)
        assert len(result) == 1
        assert result[0] == pytest.approx(99.0)

    def test_endpoints_included(self) -> None:
        """First and last values match start and stop."""
        result = _logspace(0, 999, 50)
        assert result[0] == pytest.approx(0.0, abs=1e-9)
        assert result[-1] == pytest.approx(999.0, abs=1e-6)

    def test_length_matches_num(self) -> None:
        """Output length equals requested num."""
        result = _logspace(0, 500, 20)
        assert len(result) == 20

    def test_values_monotonically_increase(self) -> None:
        """Values are strictly increasing."""
        result = _logspace(0, 1000, 100)
        for a, b in itertools.pairwise(result):
            assert b > a


# ---------------------------------------------------------------------------
# downsample_log
# ---------------------------------------------------------------------------


class TestDownsampleLog:
    """Tests for the downsample_log helper."""

    def test_passthrough_when_short(self) -> None:
        """Arrays shorter than max_points are returned unchanged."""
        arr = list(range(10))
        assert downsample_log(arr, max_points=20) == arr

    def test_passthrough_at_exact_limit(self) -> None:
        """Arrays exactly at max_points are returned unchanged."""
        arr = list(range(100))
        assert downsample_log(arr, max_points=100) == arr

    def test_reduces_length(self) -> None:
        """Long arrays are reduced to at most max_points."""
        arr = list(range(5000))
        result = downsample_log(arr, max_points=200)
        assert len(result) <= 200

    def test_preserves_first_and_last(self) -> None:
        """First and last elements of the original are preserved."""
        arr = list(range(5000))
        result = downsample_log(arr, max_points=100)
        assert result[0] == arr[0]
        assert result[-1] == arr[-1]

    def test_empty_list(self) -> None:
        """Empty input returns empty output."""
        assert downsample_log([], max_points=10) == []

    def test_single_element(self) -> None:
        """Single-element input is returned unchanged."""
        assert downsample_log([42.0], max_points=10) == [42.0]

    def test_values_are_subset_of_original(self) -> None:
        """All downsampled values exist in the original array."""
        arr = [float(i**2) for i in range(3000)]
        result = downsample_log(arr, max_points=50)
        for v in result:
            assert v in arr
