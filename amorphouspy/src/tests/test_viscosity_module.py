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

"""

import numpy as np
import pytest

from amorphouspy.workflows.viscosity import (
    auto_cutoff,
    autocorrelation_fft,
    cumulative_trapezoid,
    get_closest_divisor,
    get_viscosity,
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
    assert isinstance(out["max_lag"], list)
