"""Unit tests for the viscosity analysis module in `amorphouspy.workflows.viscosity`.

Author: Achraf Atila (achraf.atila@bam.de)

This test suite validates numerical correctness, error handling, and physical consistency
of key functions used to compute viscosity from molecular dynamics data. The tested
functionalities include:

- `autocorrelation_fft`: Verification of autocorrelation computation via FFT.
- `cumulative_trapezoid`: Integration consistency checks using the trapezoidal rule.
- `auto_cutoff`: Automated determination of cutoff times based on signal decay.
- `get_closest_divisor`: Logical correctness of divisor selection utility.
- `get_viscosity`: End-to-end validation of synthetic viscosity computation from stress tensors.
- `_extract_md_data`: Result-dict normalisation utility.
- `helfand_viscosity`: Helfand moment viscosity computation.
- `_viscosity_plateaued`: Convergence check on the viscosity integral tail.

"""

import numpy as np
import pytest
from amorphouspy.workflows.viscosity import (
    _extract_md_data,
    _viscosity_plateaued,
    auto_cutoff,
    autocorrelation_fft,
    cumulative_trapezoid,
    get_closest_divisor,
    get_viscosity,
    helfand_viscosity,
)


def test_autocorrelation_fft_identity() -> None:
    """Verify that the FFT-based autocorrelation produces finite values and a positive zero-lag peak."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    acf = autocorrelation_fft(x, max_lag=len(x))
    assert np.isfinite(acf).all()
    assert acf[0] > 0


def test_cumulative_trapezoid_basic() -> None:
    """Validate that the cumulative trapezoid integration returns a correct linear accumulation."""
    arr = np.ones(5)
    dt = 0.1
    out = cumulative_trapezoid(arr, dt)
    expected = np.array([0, 0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_auto_cutoff_noise_threshold() -> None:
    """Check that noise-threshold cutoff detection returns a valid scalar within time bounds."""
    sacf = np.array([1, 0.5, 0.1, 0.01, 0.001, 0])
    times = np.arange(len(sacf))
    t = auto_cutoff(sacf, times, method="noise_threshold", epsilon=0.05)
    assert np.isscalar(t)
    assert 0 <= float(t) <= float(times[-1])


def test_auto_cutoff_cumulative_integral() -> None:
    """Ensure that the cumulative-integral cutoff method produces a valid float cutoff time."""
    sacf = np.exp(-np.linspace(0, 5, 50))
    times = np.linspace(0, 5, 50)
    t = auto_cutoff(sacf, times, method="cumulative_integral", epsilon=0.01)
    assert isinstance(t, float)
    assert 0 <= t <= times[-1]


def test_get_closest_divisor_basic() -> None:
    """Verify correct selection of divisors closest to a given reference integer."""
    expected_1, expected_2, expected_3 = 2, 50, 100
    assert get_closest_divisor(100, 3) == expected_1
    assert get_closest_divisor(100, 50) == expected_2
    assert get_closest_divisor(100, 99) == expected_3


def test_get_closest_divisor_invalid() -> None:
    """Ensure that non-positive targets trigger ValueError in divisor selection."""
    with pytest.raises(ValueError, match="Target must be a positive integer"):
        get_closest_divisor(0, 5)


def test_get_viscosity_synthetic() -> None:
    """Test viscosity extraction on synthetic pressure tensor data for consistency and positivity."""
    n = 500
    pressures = np.zeros((n, 3, 3))
    pressures[:, 0, 1] = np.exp(-np.linspace(0, 5, n))
    pressures[:, 0, 2] = np.exp(-np.linspace(0, 4, n))
    pressures[:, 1, 2] = np.exp(-np.linspace(0, 3, n))
    result = {
        "result": {
            "pressures": pressures,
            "volume": np.ones(n) * 1e5,
            "temperature": np.ones(n) * 1000,
        }
    }

    out = get_viscosity(result, timestep=10.0, max_lag=100)
    assert "viscosity" in out
    assert out["viscosity"] > 0
    assert isinstance(out["max_lag"], (int, float, np.integer, np.floating))
    assert isinstance(out["lag_time_ps"], list)
    assert isinstance(out["sacf"], list)
    assert isinstance(out["viscosity_integral"], list)


def test_get_viscosity_max_lag_none() -> None:
    """get_viscosity with max_lag=None defaults to the full signal length."""
    n = 200
    pressures = np.zeros((n, 3, 3))
    pressures[:, 0, 1] = np.exp(-np.linspace(0, 5, n))
    pressures[:, 0, 2] = np.exp(-np.linspace(0, 4, n))
    pressures[:, 1, 2] = np.exp(-np.linspace(0, 3, n))
    result = {
        "result": {
            "pressures": pressures,
            "volume": np.ones(n) * 1e5,
            "temperature": np.ones(n) * 1000,
        }
    }
    out = get_viscosity(result, timestep=10.0, max_lag=None)
    assert "viscosity" in out


def test_get_viscosity_all_cutoff_fail() -> None:
    """When all auto_cutoff calls fail, get_viscosity raises ValueError."""
    n = 200
    pressures = np.zeros((n, 3, 3))
    pressures[:, 0, 1] = np.exp(-np.linspace(0, 5, n))
    pressures[:, 0, 2] = np.exp(-np.linspace(0, 4, n))
    pressures[:, 1, 2] = np.exp(-np.linspace(0, 3, n))
    result = {
        "result": {
            "pressures": pressures,
            "volume": np.ones(n) * 1e5,
            "temperature": np.ones(n) * 1000,
        }
    }
    with pytest.raises(ValueError, match="tau_acf detection failed"):
        get_viscosity(result, timestep=10.0, max_lag=100, cutoff_method="bogus_method")


# ---------------------------------------------------------------------------
# cumulative_trapezoid — edge cases
# ---------------------------------------------------------------------------


def test_cumulative_trapezoid_short_array() -> None:
    """Returns zeros when input is shorter than NPOINTS (2)."""
    result = cumulative_trapezoid(np.array([3.0]), dt=0.1)
    np.testing.assert_array_equal(result, np.zeros(1))


# ---------------------------------------------------------------------------
# auto_cutoff — validation errors
# ---------------------------------------------------------------------------


def test_auto_cutoff_shape_mismatch() -> None:
    """Raises ValueError when sacf and times have different shapes."""
    with pytest.raises(ValueError, match="same shape"):
        auto_cutoff(np.ones(5), np.ones(6))


def test_auto_cutoff_too_short() -> None:
    """Raises ValueError when fewer than 2 data points are supplied."""
    with pytest.raises(ValueError, match="at least 2"):
        auto_cutoff(np.array([1.0]), np.array([0.0]))


def test_auto_cutoff_non_monotonic_times() -> None:
    """Raises ValueError when times are not strictly increasing."""
    with pytest.raises(ValueError, match="strictly increasing"):
        auto_cutoff(np.ones(5), np.array([0.0, 2.0, 1.0, 3.0, 4.0]))


def test_auto_cutoff_unknown_method() -> None:
    """Raises ValueError for an unrecognised method name."""
    sacf = np.exp(-np.linspace(0, 5, 50))
    times = np.linspace(0, 5, 50)
    with pytest.raises(ValueError, match="Unknown method"):
        auto_cutoff(sacf, times, method="unknown_method")


# ---------------------------------------------------------------------------
# auto_cutoff — max_cutoff branch
# ---------------------------------------------------------------------------


def test_auto_cutoff_max_cutoff_limits_result() -> None:
    """Detected cutoff respects an explicit max_cutoff bound."""
    sacf = np.exp(-np.linspace(0, 5, 50))
    times = np.linspace(0, 5, 50)
    t = auto_cutoff(sacf, times, max_cutoff=3.0)
    assert float(t) <= 3.0


def test_auto_cutoff_max_cutoff_too_few_points() -> None:
    """Raises ValueError when max_cutoff leaves fewer than min_points."""
    sacf = np.ones(20)
    times = np.linspace(0, 5, 20)
    with pytest.raises(ValueError, match="Fewer than"):
        auto_cutoff(sacf, times, max_cutoff=0.01, min_points=10)


# ---------------------------------------------------------------------------
# auto_cutoff — zero-sacf branch (noise_threshold)
# ---------------------------------------------------------------------------


def test_auto_cutoff_zero_sacf_returns_last() -> None:
    """Returns the last time point when the SACF is identically zero."""
    sacf = np.zeros(10)
    times = np.arange(10, dtype=float)
    t = auto_cutoff(sacf, times, method="noise_threshold")
    assert float(t) == float(times[-1])


# ---------------------------------------------------------------------------
# auto_cutoff — noise_threshold: no consecutive run fallback
# ---------------------------------------------------------------------------


def test_auto_cutoff_consecutive_fallback() -> None:
    """Falls back to first below-threshold index when no consecutive run exists."""
    # Pattern: alternating below/above threshold with no 3-consecutive run.
    sacf = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    times = np.arange(len(sacf), dtype=float)
    t = auto_cutoff(sacf, times, method="noise_threshold", epsilon=0.5, consecutive=3)
    assert isinstance(float(t), float)


# ---------------------------------------------------------------------------
# auto_cutoff — cumulative_integral: zero-integral edge
# ---------------------------------------------------------------------------


def test_auto_cutoff_cumulative_integral_zero_sum() -> None:
    """Returns last time when the cumulative integral sums to zero."""
    # Alternating signal → pairwise averages are 0 → cumulative integral stays 0.
    sacf = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    times = np.arange(len(sacf), dtype=float)
    t = auto_cutoff(sacf, times, method="cumulative_integral")
    assert float(t) == float(times[-1])


# ---------------------------------------------------------------------------
# auto_cutoff — self_consistent method
# ---------------------------------------------------------------------------


def test_auto_cutoff_self_consistent_basic() -> None:
    """Self-consistent method returns a valid cutoff within the time range."""
    sacf = np.exp(-np.linspace(0, 10, 100))
    times = np.linspace(0, 10, 100)
    t = auto_cutoff(sacf, times, method="self_consistent", c=5.0)
    assert 0.0 <= float(t) <= 10.0


# ---------------------------------------------------------------------------
# auto_cutoff — return_index flag
# ---------------------------------------------------------------------------


def test_auto_cutoff_return_index_is_tuple() -> None:
    """When return_index=True, auto_cutoff returns (t_cutoff, idx)."""
    sacf = np.array([1.0, 0.5, 0.1, 0.01, 0.001, 0.0])
    times = np.arange(len(sacf), dtype=float)
    result = auto_cutoff(sacf, times, method="noise_threshold", epsilon=0.05, return_index=True)
    assert isinstance(result, tuple)
    _t_cut, idx = result
    assert 0 <= idx < len(times)


# ---------------------------------------------------------------------------
# auto_cutoff — min_cutoff constraint
# ---------------------------------------------------------------------------


def test_auto_cutoff_min_cutoff_raises_result() -> None:
    """Detected cutoff is floored to min_cutoff when it falls below it."""
    sacf = np.array([1.0, 0.5, 0.1, 0.01, 0.001, 0.0])
    times = np.arange(len(sacf), dtype=float)
    t = auto_cutoff(sacf, times, method="noise_threshold", epsilon=0.05, min_cutoff=4.0)
    assert float(t) >= 4.0


# ---------------------------------------------------------------------------
# _extract_md_data
# ---------------------------------------------------------------------------


def _make_md_inner(n: int = 50) -> dict:
    pressures = np.zeros((n, 3, 3))
    pressures[:, 0, 1] = np.exp(-np.linspace(0, 5, n))
    return {
        "pressures": pressures,
        "volume": np.ones(n) * 1e5,
        "temperature": np.ones(n) * 1000.0,
    }


def test_extract_md_data_nested_result() -> None:
    """Unwraps a {'result': inner} dict and returns inner."""
    inner = _make_md_inner()
    out = _extract_md_data({"result": inner})
    assert out is inner


def test_extract_md_data_flat() -> None:
    """Returns the dict unchanged when pressures is at the top level."""
    flat = _make_md_inner()
    out = _extract_md_data(flat)
    assert out is flat


def test_extract_md_data_missing_pressures() -> None:
    """Raises KeyError when pressures key is absent."""
    with pytest.raises(KeyError):
        _extract_md_data({"result": {"volume": [], "temperature": []}})


# ---------------------------------------------------------------------------
# helfand_viscosity
# ---------------------------------------------------------------------------


def _make_helfand_result(n: int = 600) -> dict:
    """Synthetic pressure-tensor result suitable for helfand_viscosity."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 5, n)
    pressures = np.zeros((n, 3, 3))
    for (i, j), tau in [((0, 1), 1.0), ((0, 2), 0.8), ((1, 2), 0.6)]:
        pressures[:, i, j] = np.exp(-t / tau) + 0.005 * rng.standard_normal(n)
    for k in range(3):
        pressures[:, k, k] = 0.005 * rng.standard_normal(n)
    return {
        "result": {
            "pressures": pressures,
            "volume": np.ones(n) * 1e5,
            "temperature": np.ones(n) * 1000.0,
        }
    }


def test_helfand_viscosity_returns_valid_dict() -> None:
    """helfand_viscosity returns a dict with all expected keys."""
    out = helfand_viscosity(_make_helfand_result(), timestep=10.0)
    required = {
        "viscosity",
        "viscosity_fit_residual",
        "temperature",
        "method",
        "msd",
        "lag_time_ps",
        "slope",
        "shear_modulus_inf",
        "maxwell_relaxation_time_ps",
        "bulk_viscosity",
        "mean_pressure_gpa",
    }
    assert required.issubset(out.keys())


def test_helfand_viscosity_method_tag() -> None:
    """helfand_viscosity reports method as 'helfand'."""
    out = helfand_viscosity(_make_helfand_result(), timestep=10.0)
    assert out["method"] == "helfand"


def test_helfand_viscosity_types() -> None:
    """All scalar outputs are Python floats."""
    out = helfand_viscosity(_make_helfand_result(), timestep=10.0)
    for key in ("viscosity", "viscosity_fit_residual", "temperature", "slope"):
        assert isinstance(out[key], float), f"{key} is not a float"


def test_helfand_viscosity_list_outputs() -> None:
    """MSD and lag_time_ps are lists with matching lengths."""
    out = helfand_viscosity(_make_helfand_result(), timestep=10.0)
    assert isinstance(out["msd"], list)
    assert isinstance(out["lag_time_ps"], list)
    assert len(out["msd"]) == len(out["lag_time_ps"])


def test_helfand_viscosity_explicit_max_lag() -> None:
    """helfand_viscosity respects an explicit max_lag."""
    out = helfand_viscosity(_make_helfand_result(n=600), timestep=10.0, max_lag=100)
    assert len(out["msd"]) == 100


# ---------------------------------------------------------------------------
# _viscosity_plateaued
# ---------------------------------------------------------------------------


def test_viscosity_plateaued_short_tail_returns_false() -> None:
    """Returns False when the tail contains fewer than 2 points.

    With n=5 and tail_fraction=0.2: tail_start=4, tail has 1 element → False.
    """
    arr = np.ones(5)
    assert not _viscosity_plateaued(arr, tail_fraction=0.2)


def test_viscosity_plateaued_flat_array_returns_true() -> None:
    """Returns True for an array with a flat (constant) tail."""
    arr = np.ones(100)
    assert _viscosity_plateaued(arr)


def test_viscosity_plateaued_zero_mean_returns_true() -> None:
    """Returns True when the tail mean is zero (zero-array)."""
    arr = np.zeros(100)
    assert _viscosity_plateaued(arr)


def test_viscosity_plateaued_rising_array_returns_false() -> None:
    """Returns False for an array with a steeply rising tail.

    The last 20 % ramps from 1 to 20; the relative slope exceeds rel_slope_tol.
    """
    arr = np.concatenate([np.ones(80), np.linspace(1.0, 20.0, 20)])
    assert not _viscosity_plateaued(arr)
